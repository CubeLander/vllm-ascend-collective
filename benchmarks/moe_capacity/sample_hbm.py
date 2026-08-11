# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic host-side HBM/AICore sampler.

The sampler is host-agnostic: it runs an injectable command (default
``npu-smi info``), parses the canonical Ascend ``npu-smi info`` table, and
records one CSV row per sampled device.  Hosts whose tooling differs can
inject a wrapper command that emits the same table format.  The parse
function below is the documented host-adapter boundary; it intentionally does
not embed host discovery, device probing policy, or container logic.

The loop stops when the stop file exists, tags each sample from the tag file
(so a campaign controller can mark cells in-band), and writes error rows
in-band so analysis fails closed instead of silently dropping samples.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# Summary row:  ``| 0     910B3          | OK ...``
SUMMARY_RE = re.compile(r"^\|\s*(\d+)\s+(\S+)\s+\|")
# Detail row:   ``| 0   0000:C1:00.0  | 0  0 / 0  3705 / 65536 |``
# Groups: (bus id, aicore percent, aicore-mem used/total, hbm used/total).
CHIP_RE = re.compile(r"^\|\s*\d+\s+\|\s*(\S+)\s+\|\s*([\d.]+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+\|")

CSV_FIELDS = ("utc", "monotonic_s", "tag", "device", "aicore_percent", "hbm_used_mb", "hbm_total_mb")


def parse_npu_smi_info(text: str) -> dict[int, tuple[float, int, int]]:
    """Parse one ``npu-smi info`` table into per-device readings.

    Returns ``{device: (aicore_percent, hbm_used_mb, hbm_total_mb)}``.
    """
    result: dict[int, tuple[float, int, int]] = {}
    current: int | None = None
    for line in text.splitlines():
        if match := SUMMARY_RE.match(line):
            current = int(match.group(1))
            continue
        if current is not None and (match := CHIP_RE.match(line)):
            result[current] = (float(match.group(2)), int(match.group(5)), int(match.group(6)))
    return result


def sample(command: list[str], timeout: float) -> dict[int, tuple[float, int, int]]:
    """Run the injected device-table command once and parse its stdout."""
    proc = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    return parse_npu_smi_info(proc.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", required=True, help="comma or space separated physical device ids")
    ap.add_argument("--output", type=Path, required=True, help="CSV destination")
    ap.add_argument("--stop-file", type=Path, required=True, help="loop exits once this file exists")
    ap.add_argument("--tag-file", type=Path, required=True, help="sample tag is read from this file each pass")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--command", default="npu-smi info", help="device-table command (default: npu-smi info)")
    ap.add_argument("--timeout", type=float, default=20.0, help="per-command timeout in seconds")
    args = ap.parse_args()
    devices = [int(x) for x in args.devices.replace(",", " ").split()]
    command = shlex.split(args.command)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        start = time.monotonic()
        while not args.stop_file.exists():
            tag = args.tag_file.read_text().strip() if args.tag_file.exists() else "unknown"
            now = datetime.now(timezone.utc).isoformat()
            try:
                readings = sample(command, args.timeout)
                for device in devices:
                    aicore, used, total = readings[device]
                    writer.writerow((now, round(time.monotonic() - start, 3), tag, device, aicore, used, total))
                handle.flush()
            except Exception as exc:  # keep the error in-band and let analysis fail closed
                writer.writerow((now, round(time.monotonic() - start, 3), f"ERROR:{exc}", -1, -1, -1, -1))
                handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
