"""Resource and energy instrumentation with graceful degradation.

Unavailable measurements are reported as ``None`` together with a ``*_reason``
field explaining why -- never silently zeroed or faked (avoids false precision).
"""

from __future__ import annotations

import time
from contextlib import contextmanager


def peak_cpu_rss_bytes() -> tuple[int | None, str | None]:
    """Peak resident set size. Uses stdlib ``resource`` (ru_maxrss) on Unix."""
    try:
        import resource
        import sys

        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is kilobytes on Linux, bytes on macOS.
        if sys.platform == "darwin":
            return int(val), None
        return int(val) * 1024, None
    except Exception as exc:  # pragma: no cover
        return None, f"resource module unavailable: {exc}"


class CudaMemoryProbe:
    """Peak CUDA allocated/reserved via torch, if a CUDA device is in use."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._torch = None
        if enabled:
            try:
                import torch

                if torch.cuda.is_available():
                    self._torch = torch
            except Exception:
                self._torch = None

    def reset(self) -> None:
        if self._torch is not None:
            self._torch.cuda.reset_peak_memory_stats()

    def read(self) -> dict:
        if self._torch is None:
            return {
                "peak_cuda_allocated_bytes": None,
                "peak_cuda_reserved_bytes": None,
                "cuda_reason": "CUDA/torch not available or backend is CPU/mock",
            }
        return {
            "peak_cuda_allocated_bytes": int(self._torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(self._torch.cuda.max_memory_reserved()),
            "cuda_reason": None,
        }


class EnergyMeter:
    """Per-run energy via CodeCarbon if installed; otherwise ``None`` + reason."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._tracker = None
        self._reason = None
        if enabled:
            try:
                from codecarbon import EmissionsTracker

                self._tracker = EmissionsTracker(
                    save_to_file=False, log_level="error", tracking_mode="process"
                )
            except Exception as exc:
                self._reason = f"codecarbon unavailable: {exc}"
        else:
            self._reason = "energy metering disabled"

    def start(self) -> None:
        if self._tracker is not None:
            try:
                self._tracker.start()
            except Exception as exc:  # pragma: no cover
                self._reason = f"codecarbon start failed: {exc}"
                self._tracker = None

    def stop(self) -> dict:
        if self._tracker is None:
            return {"energy_joules": None, "energy_reason": self._reason}
        try:
            emissions_kg = self._tracker.stop()
            data = getattr(self._tracker, "final_emissions_data", None)
            energy_kwh = getattr(data, "energy_consumed", None) if data else None
            joules = energy_kwh * 3.6e6 if energy_kwh is not None else None
            return {"energy_joules": joules, "energy_co2_kg": emissions_kg, "energy_reason": None}
        except Exception as exc:  # pragma: no cover
            return {"energy_joules": None, "energy_reason": f"codecarbon stop failed: {exc}"}


@contextmanager
def timer():
    t0 = time.perf_counter()
    box = {"elapsed_s": 0.0}
    try:
        yield box
    finally:
        box["elapsed_s"] = time.perf_counter() - t0
