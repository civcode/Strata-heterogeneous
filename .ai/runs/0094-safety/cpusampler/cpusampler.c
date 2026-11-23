// Handoff 0094: an on-CPU stack sampler loaded with LD_PRELOAD, because perf is not available to the user on Oberon
// (kernel.perf_event_paranoid = 4, no sudo).  Not engine code: it is a profiling tool, used for one replay.
//
// Every thread of the process gets a timer on its own CPU-time clock (499 Hz of CPU time, as `perf record -F 499`
// samples a running thread).  The SIGPROF handler records the thread id and the call stack (glibc backtrace(), which
// unwinds with .eh_frame through libraries without frame pointers; lock-free with glibc >= 2.35 / _dl_find_object)
// into a lock-free ring; a control thread drains the ring to CPUSAMPLER_DIR/samples-<pid>.bin.
// Sampling starts when CPUSAMPLER_DIR/START exists and ends when CPUSAMPLER_DIR/STOP exists (after the server is
// ready / after the replay); new threads are picked up every 250 ms.  At the end the control thread writes
// maps-<pid>.txt (for offline symbolisation) and threads-<pid>.txt (tid, name, CPU ticks).
// Build: gcc -O2 -fPIC -shared -Wall -Wextra -o libcpusampler.so cpusampler.c -lpthread
// Read: analyze.py DIR (per thread group, self and inclusive), engine_frames.py DIR (innermost engine frame).
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define DEPTH 40
#define SLOTS (1u << 18)
#define MAXT 4096
#define HZ 499

typedef struct {
    _Atomic uint64_t seq;   // index + 1 once written
    uint32_t tid, depth;
    uint64_t pc[DEPTH];
} slot_t;

static slot_t* ring;
static _Atomic uint64_t widx;
static _Atomic uint64_t dropped;
static _Atomic uint64_t ridx;
static char dir[512];
static int tids[MAXT];
static timer_t timers[MAXT];
static int ntids;
static volatile int active;

static void handler(int sig, siginfo_t* si, void* uc_) {
    (void) sig; (void) si;
    if (!active) return;
    int saved = errno;
    uint64_t i = atomic_fetch_add_explicit(&widx, 1, memory_order_relaxed);
    slot_t* s = &ring[i % SLOTS];
    if (i - atomic_load_explicit(&ridx, memory_order_relaxed) >= SLOTS - 16) {   // ring full: drop
        atomic_fetch_add_explicit(&dropped, 1, memory_order_relaxed);
        errno = saved;
        return;
    }
    void* buf[DEPTH + 3];
    int n = backtrace(buf, DEPTH + 3);
    ucontext_t* uc = (ucontext_t*) uc_;
    s->tid = (uint32_t) syscall(SYS_gettid);
    s->pc[0] = (uint64_t) uc->uc_mcontext.gregs[REG_RIP];
    // buf[0] is this handler, buf[1] the signal trampoline, buf[2] the interrupted pc (same as RIP), then callers
    int d = 1;
    for (int k = 3; k < n && d < DEPTH; ++k) s->pc[d++] = (uint64_t) buf[k];
    s->depth = (uint32_t) d;
    atomic_store_explicit(&s->seq, i + 1, memory_order_release);
    errno = saved;
}

static int known(int tid) {
    for (int k = 0; k < ntids; ++k)
        if (tids[k] == tid) return 1;
    return 0;
}

static void arm_new_threads(int self) {
    DIR* dp = opendir("/proc/self/task");
    if (!dp) return;
    struct dirent* e;
    while ((e = readdir(dp)) != NULL) {
        int tid = atoi(e->d_name);
        if (tid <= 0 || tid == self || known(tid) || ntids >= MAXT) continue;
        struct sigevent sev;
        memset(&sev, 0, sizeof sev);
        sev.sigev_notify = SIGEV_THREAD_ID;
        sev.sigev_signo = SIGPROF;
        sev._sigev_un._tid = tid;
        clockid_t clk = (clockid_t) ((~(unsigned) tid) << 3) | 6;   // MAKE_THREAD_CPUCLOCK(tid, CPUCLOCK_SCHED)
        timer_t t;
        tids[ntids] = tid;
        if (timer_create(clk, &sev, &t) != 0) { timers[ntids++] = 0; continue; }
        struct itimerspec its = {{0, 1000000000L / HZ}, {0, 1000000000L / HZ}};
        timer_settime(t, 0, &its, NULL);
        timers[ntids++] = t;
    }
    closedir(dp);
}

static int exists(const char* name) {
    char p[600];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    struct stat st;
    return stat(p, &st) == 0;
}


static void drain(int fd) {
    uint64_t w = atomic_load_explicit(&widx, memory_order_acquire);
    while (ridx < w) {
        slot_t* s = &ring[ridx % SLOTS];
        int tries = 0;
        while (atomic_load_explicit(&s->seq, memory_order_acquire) != ridx + 1 && tries < 1000) { ++tries; sched_yield(); }
        if (atomic_load_explicit(&s->seq, memory_order_acquire) == ridx + 1) {
            uint32_t hdr[2] = {s->tid, s->depth};
            if (write(fd, hdr, sizeof hdr) < 0 || write(fd, s->pc, 8u * s->depth) < 0) {}
        } else {
            atomic_fetch_add_explicit(&dropped, 1, memory_order_relaxed);
        }
        ++ridx;
    }
}

static void copy_file(const char* src, const char* dst) {
    int in = open(src, O_RDONLY), out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    char b[65536];
    ssize_t r;
    while (in >= 0 && out >= 0 && (r = read(in, b, sizeof b)) > 0)
        if (write(out, b, (size_t) r) < 0) break;
    if (in >= 0) close(in);
    if (out >= 0) close(out);
}

static void* control(void* arg) {
    (void) arg;
    sigset_t m;
    sigemptyset(&m);
    sigaddset(&m, SIGPROF);
    pthread_sigmask(SIG_BLOCK, &m, NULL);
    int self = (int) syscall(SYS_gettid);
    struct timespec ts = {0, 250000000L};
    while (!exists("START")) nanosleep(&ts, NULL);
    if (exists("STOP")) return NULL;
    char p[700];
    snprintf(p, sizeof p, "%s/samples-%d.bin", dir, (int) getpid());
    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return NULL;
    void* warm[4];
    backtrace(warm, 4);   // loads the unwinder before any signal can need it
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    active = 1;
    while (!exists("STOP")) {
        arm_new_threads(self);
        drain(fd);
        nanosleep(&ts, NULL);
    }
    active = 0;
    for (int k = 0; k < ntids; ++k)
        if (timers[k]) timer_delete(timers[k]);
    struct timespec ss = {0, 50000000L};
    nanosleep(&ss, NULL);
    drain(fd);
    close(fd);
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    snprintf(p, sizeof p, "%s/maps-%d.txt", dir, (int) getpid());
    copy_file("/proc/self/maps", p);
    snprintf(p, sizeof p, "%s/threads-%d.txt", dir, (int) getpid());
    FILE* f = fopen(p, "w");
    if (f) {
        char cmd[256] = {0};
        int cf = open("/proc/self/cmdline", O_RDONLY);
        if (cf >= 0) { ssize_t r = read(cf, cmd, sizeof cmd - 1); (void) r; close(cf); }
        fprintf(f, "# exe %s wall_s %.3f samples %llu dropped %llu hz %d\n", cmd,
                (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), (unsigned long long) atomic_load(&widx),
                (unsigned long long) atomic_load(&dropped), HZ);
        DIR* dp = opendir("/proc/self/task");
        struct dirent* e;
        while (dp && (e = readdir(dp)) != NULL) {
            int tid = atoi(e->d_name);
            if (tid <= 0) continue;
            char sp[128], name[64] = {0}, st[1024] = {0};
            snprintf(sp, sizeof sp, "/proc/self/task/%d/comm", tid);
            int c = open(sp, O_RDONLY);
            if (c >= 0) { ssize_t r = read(c, name, sizeof name - 1); (void) r; close(c); }
            name[strcspn(name, "\n")] = 0;
            snprintf(sp, sizeof sp, "/proc/self/task/%d/stat", tid);
            c = open(sp, O_RDONLY);
            if (c >= 0) { ssize_t r = read(c, st, sizeof st - 1); (void) r; close(c); }
            unsigned long ut = 0, stt = 0;
            char* q = strrchr(st, ')');
            if (q) sscanf(q + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &ut, &stt);
            fprintf(f, "%d\t%s\t%lu\t%lu\n", tid, name, ut, stt);
        }
        if (dp) closedir(dp);
        fclose(f);
    }
    return NULL;
}

__attribute__((constructor)) static void init(void) {
    const char* d = getenv("CPUSAMPLER_DIR");
    if (!d || !*d) return;
    snprintf(dir, sizeof dir, "%s", d);
    ring = mmap(NULL, sizeof(slot_t) * SLOTS, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ring == MAP_FAILED) { ring = NULL; return; }
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, NULL);
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_create(&th, &at, control, NULL);
    pthread_attr_destroy(&at);
}
