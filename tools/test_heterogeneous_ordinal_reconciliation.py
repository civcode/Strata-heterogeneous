#!/usr/bin/env python3
"""M2.8 identity reconciliation tests: synthetic JSON only, no GPU runtime."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import heterogeneous_ordinal_reconciliation as ordinal


class IdentityReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            "schema": ordinal.INVENTORY_SCHEMA, "passive_only": True,
            "runtime_initialized": False,
            "pci_graphics": [
                {"pci_bdf": "0000:04:00.0", "vendor": "nvidia", "bound_driver": "nvidia"},
                {"pci_bdf": "0000:03:00.0", "vendor": "amd", "bound_driver": "amdgpu"},
                {"pci_bdf": "0000:07:00.0", "vendor": "amd", "bound_driver": "amdgpu"},
                {"pci_bdf": "0000:7e:00.0", "vendor": "amd", "bound_driver": "amdgpu"},
            ],
            "stage_memory": [
                {"stage": 0, "backend": "cuda", "ordinal": 0},
                {"stage": 1, "backend": "hip", "ordinal": 0},
                {"stage": 2, "backend": "hip", "ordinal": 1},
            ],
        }
        self.cuda = {"schema": ordinal.PROBE_SCHEMA, "backend": "cuda",
                     "method": "runtime_device_get_pci_bus_id", "complete": True,
                     "devices": [{"ordinal": 0, "pci_bdf": "0000:04:00.0"}]}
        self.hip = {"schema": ordinal.PROBE_SCHEMA, "backend": "hip",
                    "method": "runtime_device_get_pci_bus_id", "complete": True,
                    "devices": [{"ordinal": 0, "pci_bdf": "0000:03:00.0"},
                                {"ordinal": 1, "pci_bdf": "0000:07:00.0"}]}

    def report(self, probes=None):
        return ordinal.reconcile(self.inventory,
                                 [self.cuda, self.hip] if probes is None else probes)

    def test_matching_evidence_never_certifies_gpu_ready(self):
        out = self.report()
        self.assertEqual("IDENTITY_CORRELATED_ONLY", out["readiness"])
        self.assertFalse(out["runtime_called_by_this_tool"])
        self.assertFalse(out["evidence_is_authenticated"])
        self.assertEqual([], out["blockers"])
        self.assertEqual(["0000:04:00.0", "0000:03:00.0", "0000:07:00.0"],
                         [d["runtime_reported_bdf"] for d in out["stage_mappings"]])

    def test_no_probes_fail_closed(self):
        out = self.report([])
        self.assertEqual("BLOCKED", out["readiness"])
        self.assertEqual(2, len(out["blockers"]))

    def test_one_backend_missing(self):
        self.assertIn("missing hip", str(self.report([self.cuda])["blockers"]))

    def test_runtime_ordinal_missing(self):
        self.hip["devices"].pop()
        self.assertIn("hip:1 absent", str(self.report()["blockers"]))

    def test_runtime_reported_amd_not_assumed_from_bdf_order(self):
        self.hip["devices"][0]["pci_bdf"] = "0000:7e:00.0"
        self.assertEqual("IDENTITY_CORRELATED_ONLY", self.report()["readiness"])
        self.assertEqual("0000:7e:00.0", self.report()["stage_mappings"][1]["runtime_reported_bdf"])

    def test_vendor_mismatch(self):
        self.hip["devices"][0]["pci_bdf"] = "0000:04:00.0"
        result = self.report()
        self.assertIn("vendor mismatch", str(result["blockers"]))
        self.assertIn("multiple runtime stages", str(result["blockers"]))

    def test_unlisted_bdf(self):
        self.hip["devices"][0]["pci_bdf"] = "0000:81:00.0"
        self.assertIn("absent from M2.7", str(self.report()["blockers"]))

    def test_wrong_bound_driver(self):
        self.inventory["pci_graphics"][1]["bound_driver"] = "vfio-pci"
        self.assertIn("bound-driver mismatch", str(self.report()["blockers"]))

    def test_duplicate_probe_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate identity probe"):
            self.report([self.cuda, self.cuda])

    def test_duplicate_probe_ordinals_rejected(self):
        self.hip["devices"][1]["ordinal"] = 0
        with self.assertRaisesRegex(ValueError, "repeated runtime ordinal"):
            self.report()

    def test_noncontiguous_runtime_ordinals_rejected(self):
        self.hip["devices"][1]["ordinal"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.report()

    def test_duplicate_probe_pci_rejected(self):
        self.hip["devices"][1]["pci_bdf"] = "0000:03:00.0"
        with self.assertRaisesRegex(ValueError, "PCI function in probe"):
            self.report()

    def test_invalid_bdf_rejected(self):
        self.cuda["devices"][0]["pci_bdf"] = "04:00.0"
        with self.assertRaisesRegex(ValueError, "invalid PCI BDF"):
            self.report()

    def test_probe_claims_incomplete_rejected(self):
        self.cuda["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.report()

    def test_inventory_claims_runtime_rejected(self):
        self.inventory["runtime_initialized"] = True
        with self.assertRaisesRegex(ValueError, "passive M2.7"):
            self.report()

    def test_duplicate_stage_ordinal_rejected(self):
        self.inventory["stage_memory"][2]["ordinal"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate backend:ordinal"):
            self.report()

    def test_duplicate_pci_inventory_rejected(self):
        self.inventory["pci_graphics"][2]["pci_bdf"] = "0000:03:00.0"
        with self.assertRaisesRegex(ValueError, "duplicate PCI BDF"):
            self.report()

    def test_reject_boolean_ordinal(self):
        self.cuda["devices"][0]["ordinal"] = True
        with self.assertRaisesRegex(ValueError, "invalid runtime ordinal"):
            self.report()

    def test_cli_reads_offline_only_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inv = root / "inv.json"
            cu = root / "cu.json"
            hip = root / "hip.json"
            inv.write_text(json.dumps(self.inventory))
            cu.write_text(json.dumps(self.cuda))
            hip.write_text(json.dumps(self.hip))
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                code = ordinal.main(["--inventory", str(inv), "--json"])
            self.assertEqual(1, code)
            self.assertEqual("BLOCKED", json.loads(out.getvalue())["readiness"])
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                code = ordinal.main(["--inventory", str(inv), "--probe", str(cu),
                                     "--probe", str(hip), "--json"])
            self.assertEqual(0, code)
            self.assertEqual("IDENTITY_CORRELATED_ONLY", json.loads(out.getvalue())["readiness"])

    def test_cli_fails_closed_on_bad_json(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "input.json"
            p.write_text("not json")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(2, ordinal.main(["--inventory", str(p)]))

    def test_cli_rejects_oversized_json(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "input.json"
            p.write_bytes(b"x" * (ordinal.MAX_JSON + 1))
            with redirect_stderr(io.StringIO()):
                self.assertEqual(2, ordinal.main(["--inventory", str(p)]))


if __name__ == "__main__":
    unittest.main()
