import numpy as np
import pytest

from ham.compression import vector_quant as vq


@pytest.mark.parametrize("bits", ["int8", "int4"])
def test_roundtrip_error_within_bound(bits):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    res = vq.roundtrip_error(x, bits)
    assert res["within_bound"], res
    # int8 must be strictly more accurate than int4.
    if bits == "int8":
        assert res["mean_abs_error"] < 0.05
    else:
        assert res["mean_abs_error"] < 0.4


@pytest.mark.parametrize("bits", ["int8", "int4"])
def test_serialize_deserialize_roundtrip(bits):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 32)).astype(np.float32)
    q = vq.quantize(x, bits)
    payload = vq.serialize_quantized(q)
    q2 = vq.deserialize_quantized(payload)
    assert q2.codes.shape == q.codes.shape
    np.testing.assert_array_equal(q.codes, q2.codes)
    np.testing.assert_allclose(q.dequantize(), q2.dequantize(), rtol=0, atol=1e-6)


def test_nibble_packing_roundtrip():
    codes = np.array([-8, -1, 0, 7, 3, -4, 5], dtype=np.int16)
    packed = vq.pack_nibbles(codes)
    # 7 values -> ceil(7/2) = 4 bytes.
    assert len(packed) == 4
    out = vq.unpack_nibbles(packed, len(codes))
    np.testing.assert_array_equal(out, codes)


def test_nibble_packing_rejects_out_of_range():
    with pytest.raises(ValueError):
        vq.pack_nibbles(np.array([8], dtype=np.int16))


def test_int4_smaller_than_int8_bytes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20, 48)).astype(np.float32)
    b8 = vq.quantized_nbytes(vq.quantize(x, "int8"))
    b4 = vq.quantized_nbytes(vq.quantize(x, "int4"))
    assert b4 < b8


def test_constant_row_is_safe():
    x = np.zeros((3, 8), dtype=np.float32)
    q = vq.quantize(x, "int8")
    np.testing.assert_allclose(q.dequantize(), x, atol=1e-6)
