import numpy as np

from ham.compression import serialize, text_codec


def test_text_codec_roundtrip_and_accounting():
    text = "The capital of Aurora is Verona. " * 50
    for codec in ("raw", "zlib", "auto"):
        enc = text_codec.encode(text, codec=codec)
        assert text_codec.decode(enc) == text
        assert enc.original_bytes == len(text.encode("utf-8"))
        if codec != "raw":
            assert enc.compressed_bytes < enc.original_bytes  # repetitive text compresses


def test_entropy_diagnostic_bounds():
    text = "abcdefgh" * 100
    diag = text_codec.empirical_code_length_bits(text, codec="zlib")
    assert 0.0 <= diag["order0_entropy_bits_per_byte"] <= 8.0
    assert diag["compressed_bytes"] > 0


def test_serialize_snapshot_measures_real_bytes(tmp_path):
    texts = [f"fact number {i} about entity {i}" for i in range(20)]
    embs = np.random.default_rng(0).standard_normal((20, 32)).astype(np.float32)
    meta = [{"id": i} for i in range(20)]
    acc = serialize.serialize_snapshot(
        str(tmp_path), texts, embs, meta,
        text_codec_name="zlib", vector_quant_name="int8", n_facts=20,
    )
    # Physical bytes equal the sum of files actually written.
    on_disk = sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file())
    assert acc.physical_bytes == on_disk
    # Logical float32 vectors are an upper bound vs int8-quantized physical vectors.
    assert acc.logical_vector_bytes == embs.nbytes
    assert acc.physical_vector_bytes < acc.logical_vector_bytes
    assert acc.n_facts == 20
    assert acc.bytes_per_fact == acc.physical_bytes / 20


def test_serialize_int4_smaller_than_int8(tmp_path):
    texts = [f"item {i}" for i in range(30)]
    embs = np.random.default_rng(1).standard_normal((30, 64)).astype(np.float32)
    meta = [{"id": i} for i in range(30)]
    a8 = serialize.serialize_snapshot(str(tmp_path / "i8"), texts, embs, meta,
                                      vector_quant_name="int8")
    a4 = serialize.serialize_snapshot(str(tmp_path / "i4"), texts, embs, meta,
                                      vector_quant_name="int4")
    assert a4.physical_vector_bytes < a8.physical_vector_bytes
