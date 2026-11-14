"""Explicit, passive GPU telemetry sampling with existing Strata readers.

Run separately from the bounded resident benchmark; no GPU compute or reset.
Monitor indices must be mapped to physical PCI identities on the target host.
"""
import argparse
import json
import time


def device(value):
    backend, sep, index = value.partition(":")
    if not sep or backend not in ("cuda", "hip") or not index.isascii() or not index.isdecimal():
        raise argparse.ArgumentTypeError("device must be cuda:INDEX or hip:INDEX (monitor index)")
    if len(index) > 3:
        raise argparse.ArgumentTypeError("monitor index exceeds bounds")
    return backend, int(index)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=device, action="append", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args(argv)
    if not 0.25 <= args.interval <= 5 or not 1 <= args.duration <= 3600:
        parser.error("interval must be 0.25..5 seconds and duration 1..3600 seconds")
    if len(args.device) > 16 or len(set(args.device)) != len(args.device):
        parser.error("choose at most 16 distinct monitor devices")
    from serve.telemetry import gpu_reader
    readers = [(backend, index, gpu_reader(index, amd=backend == "hip"))
               for backend, index in args.device]
    missing = [f"{backend}:{index}" for backend, index, reader in readers if not reader.ok()]
    if missing:
        parser.error("telemetry unavailable for " + ", ".join(missing))
    start = time.monotonic()
    cycle = 0
    try:
        while time.monotonic() - start < args.duration:
            sample_start = time.monotonic()
            cycle += 1
            for backend, index, reader in readers:
                read_start = time.monotonic()
                data = reader.read()
                read_end = time.monotonic()
                print(json.dumps({"unix_s": time.time(), "elapsed_s": read_end - start,
                                  "monotonic_s": read_end, "read_start_s": read_start,
                                  "cycle": cycle, "cycle_start_s": sample_start,
                                  "interval_s": args.interval,
                                  "backend": backend, "monitor_index": index,
                                  "util_percent": data.get("util"),
                                  "vram_used_bytes": data.get("mem_used"),
                                  "vram_total_bytes": data.get("mem_total"),
                                  "power_w": data.get("power"),
                                  "power_limit_w": data.get("power_limit"),
                                  "temperature_c": data.get("temp"),
                                  "pcie_gen": data.get("pcie_gen"),
                                  "pcie_width": data.get("pcie_width"),
                                  "pcie_rx_mb": data.get("pcie_rx_mb"),
                                  "pcie_tx_mb": data.get("pcie_tx_mb")}), flush=True)
            time.sleep(max(0.0, min(args.interval - (time.monotonic() - sample_start),
                                    args.duration - (time.monotonic() - start))))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
