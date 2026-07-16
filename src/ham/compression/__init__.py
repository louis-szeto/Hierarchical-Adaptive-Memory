"""Compression codecs: textual (zstd/zlib), vector quantization (int8/int4/PQ),
and physical serialization with real byte accounting."""

from . import serialize, text_codec, vector_quant

__all__ = ["serialize", "text_codec", "vector_quant"]
