import fcntl
import mmap
import os
from pathlib import Path
import stat
import tempfile
import unittest

from tools.heterogeneous_direct_graph_probe import layout, make_memfd, run, SEALS

# Separate processes exercise FD inheritance, control, sequencing, failure and
# cleanup. They emulate readiness only; they do NOT emulate a GPU or prove ordering.
FAKE = '''#!/usr/bin/env python3
import mmap, struct, sys
fd, rows, timeout, *devices = map(int, sys.argv[1:])
payload = 51216 * rows
ready = (payload + 4095) // 4096 * 4096
error, extent = ready + 64, ready + 4096
m = mmap.mmap(fd, extent)
print(f"READY {payload} {extent}", flush=True)
cuda = len(devices) == 1
state, seq = "idle", 0
for line in sys.stdin:
    parts = line.split()
    if parts == ["quit"]:
        assert state in ("idle", "poisoned")
        m.close()
        print("BYE", flush=True)
        sys.exit(0)
    op, new = parts[0], int(parts[1])
    if op in ("prepare", "arm"):
        assert state == "idle" and new > seq
        seq = new
        if cuda:
            assert op == "prepare"
            struct.pack_into("<I", m, ready, 0)
            struct.pack_into("<I", m, error, 0)
            state, out = "prepared", "PREPARED"
        else:
            assert op == "arm"
            state, out = "armed", "ARMED"
    else:
        assert new == seq
        if cuda and op == "publish":
            assert state == "prepared"
            struct.pack_into("<I", m, ready, 1)
            state, out = "launched", "QUEUED"
        elif cuda and op == "drain":
            assert state == "launched"
            state, out = "idle", "DRAINED"
        elif not cuda and op == "pending":
            assert state == "armed" and struct.unpack_from("<I", m, ready)[0] == 0
            out = "PENDING"
        elif not cuda and op == "finish":
            assert state == "armed" and struct.unpack_from("<I", m, ready)[0] == 1
            state, out = "idle", "VERIFIED"
        elif op == "reject":
            assert state == ("prepared" if cuda else "armed")
            assert struct.unpack_from("<I", m, ready)[0] == 0
            struct.pack_into("<I", m, error, 1)
            state, out = "poisoned", "REJECTED"
        else:
            raise RuntimeError(op)
    print(out, flush=True)
sys.exit(2)
'''


class DirectGraphProbeTests(unittest.TestCase):
    def test_layout_and_seals(self):
        for rows in range(1, 9):
            payload, ready, error, extent = layout(rows)
            self.assertEqual(payload, 51216 * rows)
            self.assertGreaterEqual(ready, payload)
            self.assertEqual(error, ready + 64)
            self.assertEqual(extent, ready + 4096)
            fd = make_memfd(extent)
            try:
                self.assertEqual(fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS, SEALS)
                self.assertEqual(os.fstat(fd).st_size, extent)
                with self.assertRaises(OSError):
                    os.ftruncate(fd, extent + 4096)
                with mmap.mmap(fd, extent) as page:
                    self.assertEqual(page[ready:error+4], bytes(68))
            finally:
                os.close(fd)

    def test_invalid_geometry_and_parameters(self):
        for rows in (0, 9, -1, 1.5, True):
            with self.assertRaises(ValueError):
                layout(rows)
        for opts in (dict(rounds=0), dict(rounds=2, warmup=2), dict(wait_ms=0)):
            with self.assertRaises(ValueError):
                run("no", "no", 0, [0, 1], **opts)
        with self.assertRaises(ValueError):
            run("no", "no", 0, [0, 0])

    def make_fake(self, directory, name, code=FAKE):
        path = Path(directory) / name
        path.write_text(code, encoding="utf-8")
        path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        return path

    def test_full_protocol_and_terminal_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = self.make_fake(directory, "worker")
            result = run(fake, fake, 0, [0, 1], rows=8, rounds=5, warmup=2, exercise_timeout=True)
        self.assertEqual(result["rounds_verified"], 5)
        self.assertEqual(result["withheld_publication"], "rejected")
        self.assertEqual(len(result["samples_ms"]), 3)
        self.assertIn("BYE", result["cleanup"])
        self.assertIn("model-free", result["scope"])

    def test_unexpected_response_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            good = self.make_fake(directory, "good")
            bad = self.make_fake(directory, "bad", FAKE.replace('"ARMED"', '"WRONG"'))
            with self.assertRaisesRegex(RuntimeError, "expected ARMED"):
                run(good, bad, 0, [0, 1], rounds=2, warmup=0)

    def test_bad_ready_word_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            good = self.make_fake(directory, "good")
            bad = self.make_fake(directory, "bad", FAKE.replace('m, ready, 1)', 'm, ready, 2)'))
            with self.assertRaises(RuntimeError):
                run(bad, good, 0, [0, 1], rounds=2, warmup=0)


if __name__ == "__main__":
    unittest.main()
