import unittest

from strata_control.mixed_placement import layer_plan


class MixedPlacementTests(unittest.TestCase):
    def test_qualified_plans(self):
        self.assertEqual(layer_plan("cuda:0,hip:0,hip:1", "15,33"),
                         "cuda:0:0-15,hip:0:15-33,hip:1:33-48")
        self.assertEqual(layer_plan("hip:0,hip:1", "24"),
                         "hip:0:0-24,hip:1:24-48")

    def test_refuse_invalid_or_automatic_placement(self):
        for devices, split in (
            ("hip:0,hip:1", "auto"), ("hip:0,hip:1", "0"),
            ("hip:0,hip:1", "48"), ("hip:0,hip:1", "24,33"),
            ("cuda:0,hip:0,hip:1", "33,15"), ("cuda:0,hip:0,hip:1", "15,15"),
            ("hip:0,hip:00", "24"), ("metal:0,hip:1", "24"),
            ("hip:-1,hip:1", "24"), ("hip:0,hip:1", "２４"),
        ):
            with self.subTest(devices=devices, split=split), self.assertRaises(ValueError):
                layer_plan(devices, split)

    def test_explicit_layer_count(self):
        self.assertEqual(layer_plan("cuda:0,hip:0", "4", 8), "cuda:0:0-4,hip:0:4-8")
        for invalid in (0, True, "48"):
            with self.assertRaises(ValueError):
                layer_plan("cuda:0,hip:0", "4", invalid)


if __name__ == "__main__":
    unittest.main()
