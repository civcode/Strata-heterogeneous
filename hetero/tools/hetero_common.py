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


def mem_sample():
    m = _meminfo()
    s = {"available_gib": round(m["MemAvailable"] / 2**30, 2), "shmem_gib": round(m["Shmem"] / 2**30, 2),
         "cached_gib": round(m["Cached"] / 2**30, 2)}
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
