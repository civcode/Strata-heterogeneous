"""CPU-only exhaustive small-window MTP progress and legacy protocol checks."""
import itertools
import unittest

from serve.heterogeneous_contract import Progress


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
