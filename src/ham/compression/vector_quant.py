"""Scalar (int8 / int4) and optional product quantization for stored embeddings.

Everything here is *actually applied* and produces packed bytes that round-trip,
so downstream byte accounting reflects real serialized sizes rather than estimates.
We store enough metadata (scale, zero-point, shape, dtype) to decode.

We deliberately quantize per-vector (per-row) with an affine map:

    q = round((x - zero) / scale),  x_hat = q * scale + zero

For int4, two nibbles are packed into each byte. None of this claims Shannon
optimality; it is a concrete, measurable lossy codec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LEVELS = {"int8": (8, -128, 127), "int4": (4, -8, 7)}


@dataclass
class QuantizedVectors:
    codes: np.ndarray  # int8 array (for int4, values in [-8,7] before packing)
    scale: np.ndarray  # (n,) float32 per-row scale
    zero: np.ndarray  # (n,) float32 per-row zero-point (midpoint)
    bits: int
    dim: int
    n: int

    def dequantize(self) -> np.ndarray:
        x = self.codes.astype(np.float32) * self.scale[:, None] + self.zero[:, None]
        return x.astype(np.float32)


def quantize(vectors: np.ndarray, bits: str | int = "int8") -> QuantizedVectors:
    """Affine per-row scalar quantization to int8 or int4."""
    key = bits if isinstance(bits, str) else f"int{bits}"
    if key not in _LEVELS:
        raise ValueError(f"unsupported vector_quant bits: {bits!r} (use 'int8' or 'int4')")
    nbits, qmin, qmax = _LEVELS[key]
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    n, dim = vectors.shape
    vmin = vectors.min(axis=1)
    vmax = vectors.max(axis=1)
    zero = ((vmax + vmin) / 2.0).astype(np.float32)
    span = (vmax - vmin) / 2.0
    # Avoid divide-by-zero for constant rows.
    span = np.where(span <= 1e-12, 1.0, span)
    scale = (span / float(qmax)).astype(np.float32)
    q = np.round((vectors - zero[:, None]) / scale[:, None])
    q = np.clip(q, qmin, qmax).astype(np.int8)
    return QuantizedVectors(codes=q, scale=scale, zero=zero, bits=nbits, dim=dim, n=n)


def pack_nibbles(codes: np.ndarray) -> bytes:
    """Pack an int4 code array (values in [-8, 7]) into bytes, two per byte.

    Values are biased by +8 into [0, 15] so they fit an unsigned nibble.
    The trailing nibble of an odd-length row is zero-padded; the caller must
    know ``n`` and ``dim`` to unpack exactly.
    """
    flat = np.asarray(codes, dtype=np.int16).reshape(-1)
    if flat.size and (flat.min() < -8 or flat.max() > 7):
        raise ValueError("nibble packing requires values in [-8, 7]")
    biased = (flat + 8).astype(np.uint8)
    if biased.size % 2 == 1:
        biased = np.concatenate([biased, np.zeros(1, dtype=np.uint8)])
    hi = biased[0::2]
    lo = biased[1::2]
    packed = (hi << 4) | lo
    return packed.astype(np.uint8).tobytes()


def unpack_nibbles(data: bytes, count: int) -> np.ndarray:
    """Inverse of :func:`pack_nibbles`; returns ``count`` signed values in [-8, 7]."""
    packed = np.frombuffer(data, dtype=np.uint8)
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    inter = np.empty(packed.size * 2, dtype=np.uint8)
    inter[0::2] = hi
    inter[1::2] = lo
    inter = inter[:count]
    return inter.astype(np.int16) - 8


def serialize_quantized(q: QuantizedVectors) -> dict:
    """Return a JSON/npz-friendly payload with actual packed code bytes."""
    if q.bits == 4:
        code_bytes = pack_nibbles(q.codes.reshape(-1))
    else:
        code_bytes = q.codes.astype(np.int8).tobytes()
    return {
        "bits": q.bits,
        "n": q.n,
        "dim": q.dim,
        "scale": q.scale.astype(np.float32).tobytes(),
        "zero": q.zero.astype(np.float32).tobytes(),
        "codes": code_bytes,
    }


def deserialize_quantized(payload: dict) -> QuantizedVectors:
    bits = int(payload["bits"])
    n = int(payload["n"])
    dim = int(payload["dim"])
    scale = np.frombuffer(payload["scale"], dtype=np.float32).copy()
    zero = np.frombuffer(payload["zero"], dtype=np.float32).copy()
    if bits == 4:
        flat = unpack_nibbles(payload["codes"], n * dim)
        codes = flat.astype(np.int8).reshape(n, dim)
    else:
        codes = np.frombuffer(payload["codes"], dtype=np.int8).reshape(n, dim).copy()
    return QuantizedVectors(codes=codes, scale=scale, zero=zero, bits=bits, dim=dim, n=n)


def quantized_nbytes(q: QuantizedVectors) -> int:
    """Physical bytes of the packed codes + per-row scale/zero metadata."""
    if q.bits == 4:
        code_bytes = (q.n * q.dim + 1) // 2
    else:
        code_bytes = q.n * q.dim  # int8
    meta = q.scale.nbytes + q.zero.nbytes
    return int(code_bytes + meta)


def roundtrip_error(vectors: np.ndarray, bits: str | int = "int8") -> dict:
    """Max/mean absolute reconstruction error and a theoretical per-row bound."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    q = quantize(vectors, bits)
    x_hat = q.dequantize()
    err = np.abs(vectors - x_hat)
    # Uniform-quantizer bound: half a step = scale / 2.
    bound = q.scale / 2.0
    return {
        "max_abs_error": float(err.max()) if err.size else 0.0,
        "mean_abs_error": float(err.mean()) if err.size else 0.0,
        "max_step_bound": float((q.scale).max()) if q.scale.size else 0.0,
        "within_bound": bool(np.all(err <= bound[:, None] + 1e-5)),
    }


# --- optional FAISS product quantization ------------------------------------

def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def pq_encode(vectors: np.ndarray, m: int = 8, nbits: int = 8):
    """Train a FAISS ProductQuantizer and return (codes_bytes, trained_pq).

    Raises loudly if FAISS is unavailable so a "pq" run never silently downgrades.
    """
    import faiss

    vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    n, dim = vectors.shape
    if dim % m != 0:
        raise ValueError(f"pq_subvectors={m} must divide embedding dim={dim}")
    pq = faiss.ProductQuantizer(dim, m, nbits)
    pq.train(vectors)
    codes = pq.compute_codes(vectors)
    return codes.tobytes(), pq
