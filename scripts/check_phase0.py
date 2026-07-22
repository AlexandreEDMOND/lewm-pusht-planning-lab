"""Report the Phase 0 runtime and asset checks as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def check(path: Path, label: str, required: bool, results: dict[str, object]) -> None:
    exists = path.is_file()
    results[label] = {"path": str(path), "ok": exists}
    if required and not exists:
        results["ok"] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-assets", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(os.environ["STABLEWM_HOME"])
    results: dict[str, object] = {
        "ok": True,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cache_dir": str(cache_dir),
    }
    if sys.version_info[:2] != (3, 10):
        results["ok"] = False
        results["python_requirement"] = "Python 3.10 is required."

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        results["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu": gpu_name,
        }
        if args.require_cuda and not cuda_available:
            results["ok"] = False
    except ImportError:
        results["torch"] = {"ok": False, "error": "torch is not installed"}
        results["ok"] = False

    required = args.require_assets
    check(cache_dir / "hf_pusht" / "config.json", "checkpoint_config", required, results)
    check(cache_dir / "hf_pusht" / "weights.pt", "checkpoint_weights", required, results)
    check(cache_dir / "pusht" / "lewm_object.ckpt", "evaluation_checkpoint", required, results)

    dataset_path = cache_dir / "pusht_expert_train.h5"
    check(dataset_path, "dataset", required, results)
    if dataset_path.is_file():
        try:
            import h5py

            with h5py.File(dataset_path, "r") as dataset:
                keys = sorted(dataset.keys())
            results["dataset_keys"] = keys
            missing_keys = sorted({"pixels", "action", "state"} - set(keys))
            if missing_keys:
                results["ok"] = False
                results["dataset_schema_error"] = f"Missing keys: {missing_keys}"
        except OSError as error:
            results["ok"] = False
            results["dataset_schema_error"] = str(error)

    print(json.dumps(results, indent=2, sort_keys=True))
    if not results["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
