"""Cooperative retirement of a resident coordinator and its process group."""
import os
import subprocess


def retire_process(proc, reader, timeout):
    """Caller has closed command input. Never deliver a termination signal.

    Wait for the coordinator, its output reader, and absence of its original
    process group. A zero exit code alone cannot attest that workers are gone.
    On an unresolved group or reader the caller must refuse restart/clean close.
    """
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("native coordinator did not exit; operator inspection required") from exc
    reader.join(timeout=timeout)
    if reader.is_alive():
        raise RuntimeError("native output reader did not exit; operator inspection required")
    try:
        os.killpg(proc.pid, 0)  # existence check only, not signal delivery
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise RuntimeError("cannot verify native process group exit; operator inspection required") from exc
    else:
        raise RuntimeError("native process group still exists; operator inspection required")
    proc.stdout.close()
    return code
