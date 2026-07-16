"""Run manifest: package versions, git commit, hardware, OS, config hash,
dataset/model revisions, timestamps. Written once per run for reproducibility.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from . import __version__

_PACKAGES = [
    "numpy", "scipy", "pandas", "PyYAML", "zstandard", "matplotlib",
    "torch", "transformers", "sentence-transformers", "faiss-cpu",
    "datasets", "huggingface_hub", "bitsandbytes", "psutil", "codecarbon",
]


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def build_manifest(config: dict, config_hash: str, *, dataset_revision: str | None = None,
                   model_revision: str | None = None, extra: dict | None = None) -> dict:
    packages = {name: _pkg_version(name) for name in _PACKAGES}
    manifest = {
        "harness_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": _cpu_count(),
        "packages": packages,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "config": config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None
