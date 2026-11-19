"""Shared bits of the mixed-split drivers (standard library only).

- shared_arena(): one memory object for vanilla's --shared-expert-arena, as serve/replicas.py makes it for replicas
  (a memfd, so the size of /dev/shm does not limit it).  The engines open it as /proc/<this pid>/fd/<fd>.
- mem_sample(): available RAM, shared memory and the VRAM in use per card.
- expand(): ${NAME} placeholders in a run config.
"""
import json, os, re, subprocess

ARENA_NAME = "strata-mixed-arena"


def shared_arena(inheritable=False):
    """A memfd for --shared-expert-arena; returns its /proc path.  `inheritable`: it survives an exec of this process
    (serve_mixed.py execs the server, whose pid the path names).  It lives as long as this process (or what it execs)
    keeps the descriptor; the engines' mappings keep the pages after that."""
    fd = os.memfd_create(ARENA_NAME, 0 if inheritable else os.MFD_CLOEXEC)
    if inheritable:
        os.set_inheritable(fd, True)
    return f"/proc/{os.getpid()}/fd/{fd}"


def _meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


def engine_processes():
    """pid -> {RssAnon, RssFile, RssShmem} in GiB for every running `strata` executable (the engines rename their
    threads, so the executable path decides)."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                argv0 = f.read().split(b"\0")[0].decode(errors="replace")
            if os.path.basename(argv0) != "strata":
                continue
            with open(f"/proc/{d}/status") as f:
                st = dict(l.split(":", 1) for l in f if ":" in l)
        except OSError:
            continue
        out[d] = {k: round(int(st[k].split()[0]) / 2**20, 2) for k in ("RssAnon", "RssFile", "RssShmem") if k in st}
    return out


def mem_sample():
    m = _meminfo()
    s = {"available_gib": round(m["MemAvailable"] / 2**30, 2), "shmem_gib": round(m["Shmem"] / 2**30, 2),
         "cached_gib": round(m["Cached"] / 2**30, 2), "engines": engine_processes()}
    try:
        nv = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=20).stdout.split()
        s["nvidia_used_mib"] = [int(x) for x in nv]
    except Exception as e:  # noqa: BLE001
        s["nvidia_used_mib"] = str(e)
    try:
        d = json.loads(subprocess.run(["amd-smi", "metric", "-m", "--json"], capture_output=True, text=True,
                                      timeout=30).stdout)
        s["amd_used_mb"] = [g["mem_usage"]["used_vram"]["value"] for g in d["gpu_data"]]
    except Exception as e:  # noqa: BLE001
        s["amd_used_mb"] = str(e)
    return s


def expand(value, names):
    """${NAME} -> names[NAME] or the environment; an unknown name is an error."""
    if isinstance(value, str):
        def one(m):
            k = m.group(1)
            if k in names:
                return names[k]
            if k in os.environ:
                return os.environ[k]
            raise SystemExit(f"run config: ${{{k}}} has no value (--set {k}=... or the environment)")
        return re.sub(r"\$\{([A-Za-z0-9_]+)\}", one, value)
    if isinstance(value, list):
        return [expand(v, names) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, names) for k, v in value.items() if not k.startswith("//")}
    return value


SDMA_RULE = ("HSA_ENABLE_SDMA must be 0 for every GPU process (.ai/PROJECT.md, the first rule): "
             "SDMA has a known bug in the AMD driver")


def force_sdma_off(env):
    """Sets HSA_ENABLE_SDMA=0 in `env` (a dict, or os.environ); refuses any other value already there."""
    v = env.get("HSA_ENABLE_SDMA")
    if v is not None and v != "0":
        raise SystemExit(f"HSA_ENABLE_SDMA={v}: refused. " + SDMA_RULE)
    env["HSA_ENABLE_SDMA"] = "0"
    return env


def reset_ignored_signals(for_exec=False):
    """Undo an inherited "ignore SIGINT/SIGQUIT".  A command a non-interactive shell starts in the background (`cmd &`,
    a tool runner's background job) gets both ignored, and so does everything it starts: exec keeps an ignored signal
    ignored, and a Python started that way installs no KeyboardInterrupt handler.  The server's cooperative stop
    (SIGINT, then the drain) then does nothing, and whoever waits for it waits for ever.  Every tool that starts a GPU
    process calls this first (`for_exec`: before replacing itself, the default action instead of Python's handler)."""
    import signal
    for s in (signal.SIGINT, signal.SIGQUIT):
        if signal.getsignal(s) == signal.SIG_IGN:
            signal.signal(s, signal.SIG_DFL if (for_exec or s == signal.SIGQUIT) else signal.default_int_handler)


def stop_cooperatively(p, drain_s=120, term_s=60):
    """Stop a server or engine the way it expects: SIGINT and its drain, then SIGTERM (also a cooperative stop) if it
    is still running.  Never SIGKILL (.ai/PROJECT.md).  Returns the exit status, or None when the process is still
    running after both: the caller reports it and starts no other GPU job."""
    import signal
    import subprocess
    for sig, wait_s in ((signal.SIGINT, drain_s), (signal.SIGTERM, term_s)):
        if p.poll() is not None:
            return p.returncode
        p.send_signal(sig)
        try:
            return p.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
    return p.poll()
