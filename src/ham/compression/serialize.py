"""Physically serialize a memory snapshot to disk and measure real byte sizes.

This is where "byte-honest" reporting is enforced: we write the actual store
(compressed text blob + quantized vector codes + metadata) to a directory and
read back ``os.path.getsize`` for every file. We separately report *logical*
(uncompressed, float32) bytes as an upper-bound sanity check.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

from . import text_codec, vector_quant


@dataclass
class ByteAccounting:
    logical_text_bytes: int
    logical_vector_bytes: int
    physical_text_bytes: int
    physical_vector_bytes: int
    physical_meta_bytes: int
    n_items: int
    n_facts: int
    text_codec: str
    vector_quant: str

    @property
    def logical_bytes(self) -> int:
        return self.logical_text_bytes + self.logical_vector_bytes

    @property
    def physical_bytes(self) -> int:
        return self.physical_text_bytes + self.physical_vector_bytes + self.physical_meta_bytes

    @property
    def compression_ratio(self) -> float:
        if self.physical_bytes == 0:
            return 0.0
        return self.logical_bytes / self.physical_bytes

    @property
    def bytes_per_fact(self) -> float:
        if self.n_facts == 0:
            return float("nan")
        return self.physical_bytes / self.n_facts

    def to_dict(self) -> dict:
        d = {
            "logical_text_bytes": self.logical_text_bytes,
            "logical_vector_bytes": self.logical_vector_bytes,
            "logical_bytes": self.logical_bytes,
            "physical_text_bytes": self.physical_text_bytes,
            "physical_vector_bytes": self.physical_vector_bytes,
            "physical_meta_bytes": self.physical_meta_bytes,
            "physical_bytes": self.physical_bytes,
            "n_items": self.n_items,
            "n_facts": self.n_facts,
            "compression_ratio": self.compression_ratio,
            "bytes_per_fact": self.bytes_per_fact,
            "text_codec": self.text_codec,
            "vector_quant": self.vector_quant,
        }
        return d


def _write_bytes(path: str, data: bytes) -> int:
    with open(path, "wb") as fh:
        fh.write(data)
    return os.path.getsize(path)


def serialize_snapshot(
    out_dir: str,
    texts: list[str],
    embeddings: np.ndarray | None,
    metadata: list[dict],
    *,
    text_codec_name: str = "auto",
    zstd_level: int = 10,
    vector_quant_name: str = "int8",
    n_facts: int | None = None,
) -> ByteAccounting:
    """Write ``texts`` (compressed), ``embeddings`` (quantized), and ``metadata``
    to ``out_dir`` and return actual on-disk byte accounting.

    ``vector_quant_name`` in {"none", "int8", "int4", "pq"}. "pq" requires FAISS
    and fails loudly otherwise.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_items = len(texts)

    # --- text payload: one length-prefixed blob, then compressed ------------
    blob = bytearray()
    for t in texts:
        b = t.encode("utf-8")
        blob += struct.pack("<I", len(b))
        blob += b
    logical_text_bytes = len(blob)
    # text_codec operates on str; round-trip the raw bytes through latin-1 so
    # every byte value is preserved 1:1 while reusing the same codec path.
    enc = text_codec.encode(bytes(blob).decode("latin-1"), codec=text_codec_name, level=zstd_level)
    physical_text_bytes = _write_bytes(os.path.join(out_dir, "text.blob"), enc.data)

    # --- vector payload -----------------------------------------------------
    logical_vector_bytes = 0
    physical_vector_bytes = 0
    vq_used = vector_quant_name
    vec_meta: dict = {"vector_quant": vector_quant_name}
    if embeddings is not None and len(embeddings) > 0:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        logical_vector_bytes = int(embeddings.nbytes)  # float32 logical size
        vpath = os.path.join(out_dir, "vectors.bin")
        if vector_quant_name == "none":
            physical_vector_bytes = _write_bytes(vpath, embeddings.tobytes())
        elif vector_quant_name in ("int8", "int4"):
            q = vector_quant.quantize(embeddings, vector_quant_name)
            payload = vector_quant.serialize_quantized(q)
            packed = (
                struct.pack("<III", payload["bits"], payload["n"], payload["dim"])
                + struct.pack("<I", len(payload["scale"])) + payload["scale"]
                + struct.pack("<I", len(payload["zero"])) + payload["zero"]
                + struct.pack("<I", len(payload["codes"])) + payload["codes"]
            )
            physical_vector_bytes = _write_bytes(vpath, packed)
        elif vector_quant_name == "pq":
            code_bytes, pq = vector_quant.pq_encode(embeddings)
            physical_vector_bytes = _write_bytes(vpath, code_bytes)
            vec_meta["pq_m"] = pq.M
            vec_meta["pq_nbits"] = pq.nbits
        else:
            raise ValueError(f"unknown vector_quant: {vector_quant_name!r}")

    # --- metadata -----------------------------------------------------------
    meta_payload = {
        "n_items": n_items,
        "text_codec": enc.codec,
        "vector": vec_meta,
        "items": metadata,
    }
    mpath = os.path.join(out_dir, "meta.json")
    with open(mpath, "w") as fh:
        json.dump(meta_payload, fh)
    physical_meta_bytes = os.path.getsize(mpath)

    if n_facts is None:
        n_facts = n_items

    return ByteAccounting(
        logical_text_bytes=logical_text_bytes,
        logical_vector_bytes=logical_vector_bytes,
        physical_text_bytes=physical_text_bytes,
        physical_vector_bytes=physical_vector_bytes,
        physical_meta_bytes=physical_meta_bytes,
        n_items=n_items,
        n_facts=n_facts,
        text_codec=enc.codec,
        vector_quant=vq_used,
    )
