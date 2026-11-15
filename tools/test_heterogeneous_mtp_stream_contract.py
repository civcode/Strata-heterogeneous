"""CPU-only exhaustive small-window MTP progress and legacy protocol checks."""
import itertools
import unittest

from serve.heterogeneous_contract import Progress, validate_request


def compositions(n, largest):
    if not n:
        yield ()
    else:
        for k in range(1, min(n, largest) + 1):
            for rest in compositions(n - k, largest):
                yield (k, *rest)


class MtpStreamContract(unittest.TestCase):
    def test_committed_windows_and_terminal_counts(self):
        for prompt, chunk, budget, spec in itertools.product((1, 3, 9), (1, 4, 8), range(1, 9), (1, 4)):
            for output_count in range(budget + 1):
                stopped = output_count < budget
                fed = output_count if stopped else output_count - 1
                for keeps in compositions(fed, spec + 1):
                    p = Progress(1, prompt, budget, chunk, 100, {99}, True, mtp_spec=spec)
                    ids = list(range(10, 10 + output_count))
                    before = 0
                    pos = 0
                    window = 0

                    def checkpoint(end, terminal, count):
                        nonlocal before, window
                        window += 1
                        batch = ids[before:before + count]
                        line = f"HETERO_PROGRESS 1 {window} {end} BATCH {int(terminal)} {count}"
                        if batch:
                            line += " " + " ".join(map(str, batch))
                        self.assertEqual(p.accept(line), tuple(batch))
                        before += count

                    while pos < prompt:
                        pos = min(prompt, pos + chunk)
                        checkpoint(pos, pos == prompt and output_count == 0, int(pos == prompt and output_count > 0))
                    for i, keep in enumerate(keeps):
                        terminal = stopped and i + 1 == len(keeps)
                        pos += keep
                        checkpoint(pos, terminal, keep - int(terminal))
                    self.assertEqual(p.tokens, ids)
                    self.assertEqual(p.pos, prompt + fed)
                    self.assertEqual(p.terminal, stopped)

    def test_malformed_records_leave_state_unchanged(self):
        p = Progress(7, 2, 8, 2, 100, {99}, True, mtp_spec=4)
        p.accept("HETERO_PROGRESS 7 1 2 BATCH 0 1 11")
        for line in (
            "HETERO_PROGRESS 8 2 4 BATCH 0 2 12 13",
            "HETERO_PROGRESS 7 3 4 BATCH 0 2 12 13",
            "HETERO_PROGRESS 7 2 8 BATCH 0 6 12 13 14 15 16 17",
            "HETERO_PROGRESS 7 2 4 BATCH 0 1 12",
            "HETERO_PROGRESS 7 2 4 BATCH 0 2 12 99",
            "HETERO_PROGRESS 7 2 4 BATCH 2 2 12 13",
            "HETERO_PROGRESS 7 2 4 BATCH 1 2 12 13",
            "HETERO_PROGRESS 7 2 4 BATCH 0 2 12 13 14",
            "HETERO_PROGRESS 7 2 4 12",
        ):
            with self.assertRaises(ValueError, msg=line):
                p.accept(line)
            self.assertEqual((p.pos, p.window, p.tokens, p.terminal), (2, 1, [11], False))
        self.assertEqual(p.accept("HETERO_PROGRESS 7 2 4 BATCH 0 2 12 13"), (12, 13))

    def test_bounded_bulk_progress_and_request_limit(self):
        for stream in (False, True):
            p = Progress(1, 7, 4, 8, 100, {99}, stream, native_prefill=True, mtp_spec=4)
            prefix = "HETERO_PROGRESS 1 1 6"
            suffix = " BATCH 0 0" if stream else ""
            with self.assertRaises(ValueError):
                p.accept("HETERO_PROGRESS 1 1 5" + suffix)
            self.assertEqual((p.pos, p.window), (0, 0))
            self.assertEqual(p.accept(prefix + suffix), () if stream else None)
            if stream:
                with self.assertRaises(ValueError):
                    Progress(1, 2, 4, 8, 100, {99}, True, native_prefill=True, mtp_spec=4).accept("HETERO_PROGRESS 1 1 1 BATCH 0 1 10")
            p.accept("HETERO_PROGRESS 1 2 7" + (" BATCH 0 1 11" if stream else ""))
            p.accept("HETERO_PROGRESS 1 3 10" + (" BATCH 0 3 12 13 14" if stream else ""))
            self.assertEqual(p.pos, 10)
            if stream:
                self.assertEqual(p.tokens, [11, 12, 13, 14])
        p = Progress(2, 1, 1, 8, 100, {99}, True, native_prefill=True, mtp_spec=4)
        self.assertEqual(p.accept("HETERO_PROGRESS 2 1 1 BATCH 0 1 10"), (10,))
        validate_request([1]*2047, 1, 100, 2048, mtp_bulk=True)
        validate_request([1]*2046, 2, 100, 2048, mtp_bulk=True)
        with self.assertRaises(ValueError):
            validate_request([1]*2047, 2, 100, 2048, mtp_bulk=True)
        with self.assertRaises(ValueError):
            validate_request([1]*8, 2, 100, 4096, mtp_bulk=True)
        with self.assertRaises(ValueError):
            Progress(1, 2047, 4, 8, 100, {99}, True, native_prefill=True, mtp_spec=4)
        for prompt in (8, 65, 129, 2047):
            p = Progress(1, prompt, 1, 8, 100, {99}, True, native_prefill=True, mtp_spec=4)
            self.assertEqual(p.accept(f"HETERO_PROGRESS 1 1 {prompt-1} BATCH 0 0"), ())
            self.assertEqual(p.accept(f"HETERO_PROGRESS 1 2 {prompt} BATCH 0 1 10"), (10,))

    def test_ordinary_and_buffered_protocols_unchanged(self):
        for mtp in (0, 4):
            p = Progress(1, 2, 8, 2, 100, {99}, False, mtp_spec=mtp)
            self.assertIsNone(p.accept("HETERO_PROGRESS 1 1 2"))
            with self.assertRaises(ValueError):
                p.accept("HETERO_PROGRESS 1 2 3 BATCH 0 1 10")
        p = Progress(1, 2, 2, 2, 100, {99}, True)
        self.assertEqual(p.accept("HETERO_PROGRESS 1 1 2 10"), 10)
        self.assertEqual(p.accept("HETERO_PROGRESS 1 2 3 11"), 11)
        self.assertEqual(p.tokens, [10, 11])


if __name__ == "__main__":
    unittest.main()
