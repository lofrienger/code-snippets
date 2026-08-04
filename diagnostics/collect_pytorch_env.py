#!/usr/bin/env python3
"""Print PyTorch's official system and accelerator environment report."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from torch.utils.collect_env import main as collect_environment
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "PyTorch is not installed. Install the build appropriate for this "
            "machine, then run this command again.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    collect_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
