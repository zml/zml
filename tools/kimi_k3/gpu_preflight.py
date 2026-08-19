#!/usr/bin/env python3
"""Fail closed unless an NVIDIA CUDA device is usable for Kimi K3 execution."""

from __future__ import annotations

import json
import shutil
import subprocess

import torch


def main() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    smi = subprocess.run(
        [nvidia_smi, "-L"] if nvidia_smi else ["false"],
        capture_output=True,
        text=True,
        check=False,
    )
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    report = {
        "schema_version": 1,
        "required_backend": "nvidia_gpu",
        "cpu_fallback_allowed": False,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_exit_code": smi.returncode,
        "nvidia_smi_stdout": smi.stdout.strip(),
        "nvidia_smi_stderr": smi.stderr.strip(),
        "status": "PASS" if cuda_available and device_count > 0 and smi.returncode == 0 else "BLOCKED",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
