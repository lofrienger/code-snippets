#!/usr/bin/env python3
"""Check a PyTorch installation and exercise one CUDA tensor operation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of a text report",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero exit code unless CUDA and a GPU tensor test work",
    )
    return parser.parse_args()


def collect_status(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    status: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpus": [],
        "cuda_tensor_test": "not-run",
    }

    cpu_tensor = torch.rand(2, 2)
    status["cpu_tensor_test"] = float((cpu_tensor @ cpu_tensor).sum().item())

    if not cuda_available:
        return status

    for device_id in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_id)
        status["gpus"].append(
            {
                "id": device_id,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_gib": round(properties.total_memory / 1024**3, 2),
            }
        )

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = (gpu_tensor.square().sum()).item()
        torch.cuda.synchronize(device)
        status["cuda_tensor_test"] = "passed"
        status["cuda_tensor_result"] = float(result)
    except (RuntimeError, AssertionError) as exc:
        status["cuda_tensor_test"] = "failed"
        status["cuda_tensor_error"] = str(exc)

    return status


def print_text_report(status: dict[str, Any]) -> None:
    print(f"Python: {status['python_version']}")
    print(f"Platform: {status['platform']}")
    print(f"PyTorch: {status['pytorch_version']}")
    print(f"CUDA used to build PyTorch: {status['cuda_build_version'] or 'none'}")
    print(f"CUDA available: {status['cuda_available']}")
    print(f"cuDNN available: {status['cudnn_available']}")
    print(f"cuDNN version: {status['cudnn_version'] or 'none'}")
    print(f"GPU count: {status['gpu_count']}")
    for gpu in status["gpus"]:
        print(
            "GPU {id}: {name}; compute capability {compute_capability}; "
            "{total_memory_gib:.2f} GiB".format(**gpu)
        )
    print(f"CPU tensor test: passed ({status['cpu_tensor_test']:.6f})")
    print(f"CUDA tensor test: {status['cuda_tensor_test']}")
    if "cuda_tensor_error" in status:
        print(f"CUDA error: {status['cuda_tensor_error']}")


def main() -> int:
    args = parse_args()
    try:
        import torch
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        error = {"pytorch_import": "failed", "error": str(exc)}
        if args.json:
            print(json.dumps(error, ensure_ascii=False, indent=2))
        else:
            print("PyTorch import failed.", file=sys.stderr)
            print(str(exc), file=sys.stderr)
        return 2

    status = collect_status(torch)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_text_report(status)

    if args.require_cuda and (
        not status["cuda_available"] or status["cuda_tensor_test"] != "passed"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
