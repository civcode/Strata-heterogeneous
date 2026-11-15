#!/usr/bin/env python3
"""Qualify CUDA0 -> HIP0 -> HIP1 via unchanged captured native handoff kernels.

This is a model-free probe. Timings include IPC, graph launch, verification
readback and drains; they are NOT model throughput or GPU-only timings.
"""
import argparse
import fcntl
import json
import mmap
import os
from pathlib import Path
import select
import statistics
import struct
import subprocess
import time

HC, EMBED, ALIGN, MAX_SEQUENCE = 4, 2560, 4096, 65536
SEALS = fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


def layout(rows):
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 8:
        raise ValueError("rows must be an integer in 1..8")
    payload = (HC * EMBED + EMBED + HC) * rows * 4
    ready = (payload + ALIGN - 1) // ALIGN * ALIGN
    return payload, ready, ready + 64, ready + ALIGN


def make_memfd(extent):
    fd = os.memfd_create("strata-direct-graph-probe", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(fd, extent)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
        return fd
    except BaseException:
        os.close(fd)
        raise


class Worker:
    def __init__(self, executable, fd, rows, wait_ms, devices):
        args = [str(executable), str(fd), str(rows), str(wait_ms), *map(str, devices)]
        self.process = subprocess.Popen(args, pass_fds=(fd,), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, bufsize=0)
        self.buf = b""

    def expect(self, response, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise RuntimeError(f"worker {self.process.pid}: timeout waiting for {response}")
            data = os.read(self.process.stdout.fileno(), 4096)
            if not data:
                raise RuntimeError(f"worker {self.process.pid}: EOF waiting for {response}")
            self.buf += data
            if len(self.buf) > 16384:
                raise RuntimeError("oversized worker response")
        line, self.buf = self.buf.split(b"\n", 1)
        if line != response.encode():
            raise RuntimeError(f"worker {self.process.pid}: expected {response}; got {line!r}")

    def command(self, command, response, timeout):
        # One small command per window; there is never an outstanding second command.
        data = (command + "\n").encode()
        if self.process.stdin.write(data) != len(data):
            raise RuntimeError("short graph probe command")
        self.expect(response, timeout)

    def close_input(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()

    def close_output(self):
        if self.process.stdout and not self.process.stdout.closed:
            self.process.stdout.close()


def words(page, ready, error):
    return struct.unpack_from("<I", page, ready)[0], struct.unpack_from("<I", page, error)[0]


def require_words(page, ready, error, expected):
    actual = words(page, ready, error)
    if actual != expected:
        raise RuntimeError(f"cross-edge ready/error: expected {expected}; observed {actual}")


def run(cuda, hip, cuda_device, hip_devices, rows=8, rounds=128, warmup=8,
        wait_ms=2000, exercise_timeout=False, worker_factory=Worker):
    payload, ready, error, extent = layout(rows)
    if (not 1 <= rounds <= MAX_SEQUENCE - 1 or not 0 <= warmup < rounds or
            not 100 <= wait_ms <= 15000 or cuda_device < 0 or len(hip_devices) != 2 or
            min(hip_devices) < 0 or len(set(hip_devices)) != 2):
        raise ValueError("invalid rounds, warmup, timeout or runtime-local ordinals")
    # Graph launch/driver timeouts are separate. On failure only EOF is sent;
    # a stalled native driver is never forcibly killed by the probe.
    timeout = max(15.0, wait_ms / 1000 + 10.0)
    fd = make_memfd(extent)
    workers = []
    elapsed_ms = []
    try:
        with mmap.mmap(fd, extent, access=mmap.ACCESS_WRITE) as page:
            for exe, devices in ((cuda, [cuda_device]), (hip, list(hip_devices))):
                workers.append(worker_factory(exe, fd, rows, wait_ms, devices))
            c, h = workers
            for worker in workers:
                worker.expect(f"READY {payload} {extent}", timeout)
            for sequence in range(1, rounds + 1):
                # No CPU payload write or read between publication and consumption.
                c.command(f"prepare {sequence}", "PREPARED", timeout)
                require_words(page, ready, error, (0, 0))
                h.command(f"arm {sequence}", "ARMED", timeout)
                h.command(f"pending {sequence}", "PENDING", timeout)
                require_words(page, ready, error, (0, 0))
                start = time.perf_counter_ns()
                c.command(f"publish {sequence}", "QUEUED", timeout)
                # HIP0 and HIP1 are already waiting on their native GPU flags.
                # No CPU signaling of ready; do not drain producer before receiver.
                h.command(f"finish {sequence}", "VERIFIED", timeout)
                c.command(f"drain {sequence}", "DRAINED", timeout)
                require_words(page, ready, error, (1, 0))
                if sequence > warmup:
                    elapsed_ms.append((time.perf_counter_ns() - start) / 1e6)
            if exercise_timeout:
                sequence = rounds + 1
                c.command(f"prepare {sequence}", "PREPARED", timeout)
                h.command(f"arm {sequence}", "ARMED", timeout)
                h.command(f"pending {sequence}", "PENDING", timeout)
                require_words(page, ready, error, (0, 0))
                # Deliberately withhold producer publication. Native waits are
                # bounded; neither side may reuse the poisoned graph session.
                h.command(f"reject {sequence}", "REJECTED", timeout)
                require_words(page, ready, error, (0, 1))
                c.command(f"reject {sequence}", "REJECTED", timeout)
            for worker in workers:
                worker.command("quit", "BYE", timeout)
                if worker.process.wait(timeout=timeout) != 0:
                    raise RuntimeError("direct graph worker cleanup failed")
    finally:
        for worker in workers:
            worker.close_input()
        for worker in workers:
            try:
                worker.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"UNVERIFIED cleanup: worker PID {worker.process.pid} still alive; no forced kill", flush=True)
            finally:
                worker.close_output()
        os.close(fd)
    return {"schema": "strata-direct-graph-probe/v1", "status": "complete",
            "scope": "model-free; unchanged native kernels, captured GPU graphs, no CPU payload relay",
            "rows": rows, "payload_bytes_per_edge": payload, "rounds_verified": rounds,
            "warmup_excluded": warmup, "withheld_publication": "rejected" if exercise_timeout else "NOT_RUN",
            "cleanup": "BYE and exit 0 from both workers",
            "timing_source": "host publish command through both verifies and drains; includes IPC/readback",
            "median_ms": statistics.median(elapsed_ms), "min_ms": min(elapsed_ms),
            "max_ms": max(elapsed_ms), "samples_ms": elapsed_ms}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda-worker", required=True, type=Path)
    p.add_argument("--hip-worker", required=True, type=Path)
    p.add_argument("--cuda-device", required=True, type=int)
    p.add_argument("--hip-devices", required=True, nargs=2, type=int)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--rounds", type=int, default=128)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--wait-ms", type=int, default=2000)
    p.add_argument("--exercise-timeout", action="store_true")
    p.add_argument("--output", type=Path, help="new private JSON file; existing output is never overwritten")
    a = p.parse_args(argv)
    for executable in (a.cuda_worker, a.hip_worker):
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            p.error("use absolute paths to existing executable probes")
    if a.output and a.output.exists():
        p.error("output already exists")
    result = run(a.cuda_worker, a.hip_worker, a.cuda_device, a.hip_devices, a.rows,
                 a.rounds, a.warmup, a.wait_ms, a.exercise_timeout)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if a.output:
        fd = os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
