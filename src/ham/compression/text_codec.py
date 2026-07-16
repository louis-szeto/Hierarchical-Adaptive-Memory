"""Textual payload codecs for on-disk stored memory text.

We compress the *bytes actually written to disk*. ``zstd`` is used when the
optional ``zstandard`` dependency is present; otherwise we fall back to the
stdlib ``zlib`` codec so the harness is fully functional without extra deps.
The codec name that was actually used is recorded so byte accounting is honest.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass


def zstd_available() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class EncodedText:
    codec: str
    data: bytes
    original_bytes: int

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)

    @property
    def ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 0.0
        return self.original_bytes / self.compressed_bytes


def _resolve(codec: str) -> str:
    if codec == "auto":
        return "zstd" if zstd_available() else "zlib"
    if codec == "zstd" and not zstd_available():
        raise RuntimeError(
            "text_codec='zstd' requested but the 'zstandard' package is not installed; "
            "install the [zstd] extra or set text_codec='auto'/'zlib'."
        )
    return codec


def encode(text: str, codec: str = "auto", level: int = 10) -> EncodedText:
    resolved = _resolve(codec)
    raw = text.encode("utf-8")
    if resolved == "raw":
        return EncodedText("raw", raw, len(raw))
    if resolved == "zlib":
        return EncodedText("zlib", zlib.compress(raw, min(level, 9)), len(raw))
    if resolved == "zstd":
        import zstandard as zstd

        comp = zstd.ZstdCompressor(level=level)
        return EncodedText("zstd", comp.compress(raw), len(raw))
    raise ValueError(f"unknown text codec: {codec!r}")


def decode(enc: EncodedText) -> str:
    if enc.codec == "raw":
        return enc.data.decode("utf-8")
    if enc.codec == "zlib":
        return zlib.decompress(enc.data).decode("utf-8")
    if enc.codec == "zstd":
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(enc.data).decode("utf-8")
    raise ValueError(f"unknown text codec: {enc.codec!r}")


def shannon_entropy_bits_per_byte(data: bytes) -> float:
    """Empirical order-0 Shannon entropy H = -sum p log2 p over byte values.

    This is a *diagnostic lower-bound reference* for an order-0 model, not a
    claim of optimality (the Kolmogorov-optimal length is uncomputable).
    """
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def empirical_code_length_bits(text: str, codec: str = "auto", level: int = 10) -> dict:
    """Compare the order-0 entropy floor to what the real codec achieves."""
    enc = encode(text, codec=codec, level=level)
    raw = text.encode("utf-8")
    h = shannon_entropy_bits_per_byte(raw)
    return {
        "codec": enc.codec,
        "original_bytes": enc.original_bytes,
        "compressed_bytes": enc.compressed_bytes,
        "ratio": enc.ratio,
        "order0_entropy_bits_per_byte": h,
        "order0_entropy_floor_bytes": (h * len(raw)) / 8.0 if raw else 0.0,
        "achieved_bits_per_byte": (enc.compressed_bytes * 8.0 / len(raw)) if raw else 0.0,
    }
