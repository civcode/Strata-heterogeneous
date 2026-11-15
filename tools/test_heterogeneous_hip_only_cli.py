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

    def test_explicit_pipeline_mode(self):
        args = Namespace(coordinator=sys.executable, plan=HIP, fingerprint=1,
                         layers=48, hc=4, n_embd=2560, timeout_ms=1000,
                         hip_worker=sys.executable, config=[], shared=True,
                         max_new_tokens=8, prompt_chunk=1, max_context=2048,
                         group_native_hip=True, stream_tokens=True, pipeline_mode="fast")
        with patch("strata_cli.heterogeneous._native", return_value={"kv":"int8"}) as native:
            with self.assertRaisesRegex(ValueError, "lacks explicit pipeline-mode"):
                prepare_native(args)
            args.shared = False
            with self.assertRaisesRegex(ValueError, "pipeline mode requires"):
                prepare_native(args)
            self.assertEqual(native.call_count, 1)
        cfg = {"heterogeneous": dict(plan=HIP, coordinator=sys.executable,
                   hip_worker=sys.executable, config=["hip0.cfg", "hip1.cfg"],
                   fingerprint=1, pipeline_mode="fast")}
        with patch("strata_cli.heterogeneous.prepare_native",
                   return_value=(["coordinator"], {}, None, {1})), \
             patch("strata_cli.heterogeneous.model_template", return_value=None), \
             patch("strata_cli.heterogeneous._native"):
            self.assertEqual(prepare_server(cfg)[0].pipeline_mode, "fast")
            cfg["heterogeneous"]["pipeline_mode"] = "unknown"
            with self.assertRaisesRegex(ValueError, "heterogeneous.pipeline_mode"):
                prepare_server(cfg)

    def test_mtp_service_refuses_incompatible_settings_and_stale_coordinator(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as rt:
            args = Namespace(coordinator=sys.executable, plan="cuda:0:0-15,hip:0:15-33,hip:1:33-48", fingerprint=1,
                             layers=48, hc=4, n_embd=2560, timeout_ms=1000,
                             cuda_worker=sys.executable, hip_worker=sys.executable, config=[], shared=True,
                             max_new_tokens=8, prompt_chunk=1, max_context=2048, group_native_hip=True,
                             mtp=rt, spec=4, spec_min_p=0.5)
            with patch("strata_cli.heterogeneous._native",return_value={"kv":"int8"}) as native:
                with self.assertRaisesRegex(ValueError,"lacks request-local MTP service"):
                    prepare_native(args)
                for key,value in (("max_context",262144),("native_prefill_chunk",128),("spec",3),("spec_min_p",0.1)):
                    old=getattr(args,key,None)
                    setattr(args,key,value)
                    with self.assertRaisesRegex(ValueError,"MTP service candidate"):
                        prepare_native(args)
                    if old is None: delattr(args,key)
                    else: setattr(args,key,old)
                self.assertEqual(native.call_count,1)
                args.stream_tokens=True
                native.return_value["mtp_service_cap"]=1
                with self.assertRaisesRegex(ValueError,"lacks MTP streaming capability"):
                    prepare_native(args)

    def test_hip_only_mtp_needs_matching_capability(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as rt:
            args = Namespace(coordinator=sys.executable, plan=HIP, fingerprint=1,
                             layers=48, hc=4, n_embd=2560, timeout_ms=1000,
                             hip_worker=sys.executable, config=[], shared=True,
                             max_new_tokens=32, prompt_chunk=8, max_context=2048,
                             group_native_hip=True, mtp=rt, spec=4, spec_min_p=0.5)
            with patch("strata_cli.heterogeneous._native", return_value={"kv":"int8","mtp_service_cap":1}) as native:
                with self.assertRaisesRegex(ValueError,"lacks HIP-only MTP capability"):
                    prepare_native(args)
                self.assertNotIn("--cuda-worker", native.call_args.args[0])
                native.return_value["mtp_hip_pair_cap"] = 1
                with self.assertRaisesRegex(ValueError,"invalid validated model source"):
                    prepare_native(args)
                args.group_native_hip = False
                with self.assertRaisesRegex(ValueError,"MTP service candidate"):
                    prepare_native(args)

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
