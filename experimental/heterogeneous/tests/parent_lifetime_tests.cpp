#include "strata_heterogeneous/parent_lifetime.hpp"
#include <cassert>
#include <sys/wait.h>

int main() {
    using strata::heterogeneous::bind_worker_lifetime;
    bool rejected = false;
    try { bind_worker_lifetime("not-a-pid"); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    assert(::prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0);
    int ready[2]; assert(::pipe(ready) == 0);
    const pid_t coordinator = ::fork(); assert(coordinator >= 0);
    if (coordinator == 0) {
        const pid_t me = ::getpid();
        const pid_t worker = ::fork();
        if (worker < 0) _exit(2);
        if (worker == 0) {
            ::close(ready[0]);
            bind_worker_lifetime(std::to_string(me));
            const pid_t self = ::getpid();
            if (::write(ready[1], &self, sizeof(self)) != sizeof(self)) _exit(3);
            for (;;) ::pause();
        }
        ::close(ready[1]);
        pid_t armed = -1;
        if (::read(ready[0], &armed, sizeof(armed)) != sizeof(armed) || armed != worker) _exit(4);
        _exit(0);
    }
    ::close(ready[0]); ::close(ready[1]);
    int status = 0;
    assert(::waitpid(coordinator, &status, 0) == coordinator);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(::waitpid(-1, &status, 0) > 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
}
