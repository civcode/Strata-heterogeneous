"""CPU argument and server admission tests, with native/model calls mocked."""
from argparse import Namespace
import sys
import unittest
from unittest.mock import patch
from serve.heterogeneous import prepare_server
from strata_cli.heterogeneous import worker_arguments, kv_argument, prepare_native

HIP = "hip:0:0-24,hip:1:24-48"
MIXED = "cuda:0:0-12,hip:0:12-30,hip:1:30-48"


class HipOnlyCLI(unittest.TestCase):
    def test_only_planned_executable_is_required(self):
        args = Namespace(plan=HIP, hip_worker=sys.executable,
                         cuda_worker="/nonexistent/cuda-worker")
        result = worker_arguments(args)
        self.assertEqual(result[0::2], ["--hip-worker"])
        self.assertTrue(result[1])
        args.plan = MIXED
        args.cuda_worker = None
        with self.assertRaisesRegex(ValueError, "cuda-worker"):
            worker_arguments(args)
        args.cuda_worker = sys.executable
        self.assertEqual(worker_arguments(args)[0::2], ["--cuda-worker", "--hip-worker"])

    def test_missing_hip_and_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "hip-worker"):
            worker_arguments(Namespace(plan=HIP))
        with self.assertRaises(ValueError):
            worker_arguments(Namespace(plan="other:0:0-48"))

    def test_kv_mode(self):
        self.assertEqual(kv_argument(Namespace()), "int8")
        self.assertEqual(kv_argument(Namespace(kv="f16")), "f16")
        for bad in ("q4", "Q8", "", True, None):
            with self.assertRaisesRegex(ValueError, "KV mode"):
                kv_argument(Namespace(kv=bad))

    def test_native_kv_argument_and_attestation(self):
        args = Namespace(coordinator=sys.executable, plan=HIP, fingerprint=1,
                         layers=48, hc=4, n_embd=2560, timeout_ms=1000,
                         hip_worker=sys.executable, config=[], shared=True,
                         max_new_tokens=8, prompt_chunk=1, max_context=2048)
        for mode in ("int8", "f16"):
            args.kv = mode
            with patch("strata_cli.heterogeneous._native", return_value={"kv": "wrong"}) as native:
                with self.assertRaisesRegex(ValueError, "did not attest"):
                    prepare_native(args)
                argv = native.call_args.args[0]
                self.assertEqual(argv[argv.index("--kv") + 1], mode)

    def test_server_without_cuda_worker(self):
        cfg = {"heterogeneous": dict(plan=HIP, coordinator=sys.executable,
                   hip_worker=sys.executable, config=["hip0.cfg", "hip1.cfg"],
                   fingerprint=1, workload_mode=True, group_native_hip=True)}
        with patch("strata_cli.heterogeneous.prepare_native",
                   return_value=(["coordinator"], {}, None, {1})) as native, \
             patch("strata_cli.heterogeneous.model_template", return_value=None), \
             patch("strata_cli.heterogeneous._native") as preflight:
            args, argv, *_ = prepare_server(cfg)
            self.assertIsNone(args.cuda_worker)
            self.assertEqual(args.kv, "int8")
            self.assertEqual(argv, ["coordinator", "--resident"])
            native.assert_called_once_with(args)
            preflight.assert_called_once_with(argv)
            cfg["heterogeneous"]["plan"] = MIXED
            with self.assertRaisesRegex(ValueError, "cuda_worker"):
                prepare_server(cfg)
            self.assertEqual(native.call_count, 1)
            cfg["heterogeneous"]["plan"] = HIP
            cfg["heterogeneous"]["kv"] = "f16"
            self.assertEqual(prepare_server(cfg)[0].kv, "f16")
            cfg["heterogeneous"]["kv"] = "q4"
            with self.assertRaisesRegex(ValueError, "heterogeneous.kv"):
                prepare_server(cfg)
            self.assertEqual(native.call_count, 2)


if __name__ == "__main__":
    unittest.main()
