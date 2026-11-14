#!/usr/bin/env python3
"""Explicit model-free CUDA/HIP native mapped-handoff qualification. Linux only."""
import argparse
import mmap
import os
import select
import struct
import subprocess
import time

SIZE = 65536
WORDS = 4096
FLAG = 32768
ERROR = FLAG + 64


class Worker:
    def __init__(self, exe, fd, devices):
        self.p = subprocess.Popen([exe, str(fd), *map(str, devices)],
                                  pass_fds=(fd,), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, bufsize=0)
        self.buf = b""

    def expect(self, expected, timeout=15):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self.p.stdout], [], [], left)[0]:
                raise RuntimeError(f"worker {self.p.pid}: timeout waiting for {expected}")
            chunk = os.read(self.p.stdout.fileno(), 4096)
            if not chunk:
                raise RuntimeError(f"worker {self.p.pid}: EOF waiting for {expected}")
            self.buf += chunk
            if len(self.buf) > 16384:
                raise RuntimeError("oversized worker response")
        line, self.buf = self.buf.split(b"\n", 1)
        if line.decode() != expected:
            raise RuntimeError(f"worker {self.p.pid}: expected {expected}, got {line!r}")

    def command(self, cmd, response):
        self.p.stdin.write((cmd + "\n").encode())
        self.expect(response)


def run(cuda, hip, cuda_device, hip_devices, rounds):
    if not 1 <= rounds <= 32 or cuda_device < 0 or len(hip_devices) != 2 or len(set(hip_devices)) != 2 or min(hip_devices) < 0:
        raise ValueError("invalid rounds or runtime-local ordinals")
    fd = os.memfd_create("strata-overlap-probe", os.MFD_CLOEXEC)
    workers = []
    try:
        os.ftruncate(fd, SIZE)
        with mmap.mmap(fd, SIZE, access=mmap.ACCESS_WRITE) as page:
            for exe, devices in ((cuda, [cuda_device]), (hip, hip_devices)):
                w = Worker(exe, fd, devices)
                workers.append(w)
            for w in workers:
                w.expect("READY")
            c, h = workers
            c0 = (c, cuda_device)
            h0, h1 = [(h, x) for x in hip_devices]
            sequence = 0
            for name, sender, receiver in (("CUDA->HIP0", c0, h0), ("CUDA->HIP1", c0, h1),
                                           ("HIP0->CUDA", h0, c0), ("HIP1->CUDA", h1, c0),
                                           ("HIP0->HIP1", h0, h1)):
                for _ in range(rounds):
                    sequence += 1
                    page[:] = bytes(SIZE)  # both previous streams are drained
                    sw, sd = sender
                    rw, rd = receiver
                    sw.command(f"prepare {sd} {sequence}", "PREPARED")
                    rw.command(f"arm {rd} {sequence}", "ARMED")
                    time.sleep(0.010)
                    rw.command(f"pending {rd} {sequence}", "PENDING")
                    if struct.unpack_from("<I", page, FLAG)[0] != 0:
                        raise RuntimeError("ready word rose before producer launch")
                    sw.command(f"publish {sd} {sequence}", "QUEUED")
                    # No producer drain/host signal before consumer completion.
                    rw.command(f"finish {rd} {sequence}", "VERIFIED")
                    sw.command(f"drain {sd} {sequence}", "DRAINED")
                    if struct.unpack_from("<I", page, FLAG)[0] != 1 or struct.unpack_from("<I", page, ERROR)[0] != 0:
                        raise RuntimeError("invalid ready/error words after drain")
                print(f"PASS {name}: {rounds} real publish/wait/readback rounds", flush=True)
            for w in workers:
                w.command("quit", "BYE")
                if w.p.wait(timeout=15) != 0:
                    raise RuntimeError("worker cleanup failed")
    finally:
        # EOF requests normal cleanup; never kill GPU processes if a driver stalls.
        for w in workers:
            if w.p.stdin:
                w.p.stdin.close()
        for w in workers:
            try:
                w.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print(f"UNVERIFIED cleanup: worker PID {w.p.pid} still alive; no forced kill", flush=True)
            if w.p.stdout:
                w.p.stdout.close()
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda-worker", required=True)
    p.add_argument("--hip-worker", required=True)
    p.add_argument("--cuda-device", required=True, type=int)
    p.add_argument("--hip-devices", required=True, type=int, nargs=2)
    p.add_argument("--rounds", type=int, default=8)
    a = p.parse_args()
    run(a.cuda_worker, a.hip_worker, a.cuda_device, a.hip_devices, a.rounds)


if __name__ == "__main__":
    main()
