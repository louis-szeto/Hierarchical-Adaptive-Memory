import numpy as np

from ham.memory.consolidation import Consolidator
from ham.memory.store import SEMANTIC, MemoryRecord, MemoryStore


def _rec(store, vec, text="t"):
    return MemoryRecord(id=store.new_id(), text=text,
                        embedding=np.asarray(vec, dtype=np.float32))


def test_similar_records_merge_into_one_prototype():
    store = MemoryStore()
    con = Consolidator(store, radius=0.25)
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(5):
        v = base + np.array([0, 0.01 * i, 0, 0], dtype=np.float32)
        con.consolidate(_rec(store, v, text=f"near {i}"), now=i)
    assert con.prototype_count() == 1
    proto = con.prototypes[0]
    assert proto.is_prototype and proto.tier == SEMANTIC
    assert len(proto.members) == 5
    assert proto.stability > 0.6  # grows with confirmations


def test_dissimilar_records_make_separate_prototypes():
    store = MemoryStore()
    con = Consolidator(store, radius=0.25)
    con.consolidate(_rec(store, [1, 0, 0, 0]), now=0)
    con.consolidate(_rec(store, [0, 1, 0, 0]), now=1)
    con.consolidate(_rec(store, [0, 0, 1, 0]), now=2)
    assert con.prototype_count() == 3


def test_prototype_keeps_longest_exemplar_text():
    store = MemoryStore()
    con = Consolidator(store, radius=0.5)
    con.consolidate(_rec(store, [1, 0, 0, 0], text="short"), now=0)
    con.consolidate(_rec(store, [1, 0.01, 0, 0], text="a much longer exemplar text"), now=1)
    assert con.prototypes[0].text == "a much longer exemplar text"
