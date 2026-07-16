"""Deterministic synthetic multi-session fact-recall benchmark.

Generated locally with a fixed seed (no downloads, contamination-free, fully
reproducible -- RULER-style). Each example is a multi-session history in which
one target fact of the form "The <attribute> of <entity> is <value>." is stated
in some session, surrounded by distractor facts in other sessions. A subset of
examples exercise *knowledge updates* (the fact is restated with a new value in
a later session; the gold answer is the latest value), mirroring LongMemEval's
knowledge-update category.
"""

from __future__ import annotations

import random

from .base import DatasetAdapter, Example, Turn

_ENTITIES = [
    "Aurora", "Basalt", "Cobalt", "Delphi", "Ember", "Flint", "Groveton",
    "Harbor", "Ionia", "Juniper", "Kestrel", "Lumen", "Marlowe", "Nimbus",
    "Onyx", "Pinnacle", "Quartz", "Riverton", "Sable", "Thistle",
]
_ATTRIBUTES = [
    ("capital", ["Verona", "Aldgate", "Cresthaven", "Windmere", "Ashford", "Dunmore"]),
    ("mascot", ["falcon", "otter", "lynx", "heron", "ibex", "marten"]),
    ("founding year", ["1887", "1902", "1935", "1968", "1991", "2004"]),
    ("signature dish", ["saffron stew", "plum tart", "cedar bread", "kelp soup", "fig relish"]),
    ("annual festival", ["Lantern Days", "Tide Fair", "Emberfest", "Harvest Vigil", "Frost Gala"]),
    ("official color", ["teal", "amber", "crimson", "indigo", "olive", "magenta"]),
]


class SyntheticAdapter(DatasetAdapter):
    name = "synthetic"

    def __init__(self, num_examples: int = 12, num_sessions: int = 5,
                 facts_per_session: int = 4, distractors_per_session: int = 3,
                 seed: int = 0):
        self.num_examples = num_examples
        self.num_sessions = num_sessions
        self.facts_per_session = facts_per_session
        self.distractors_per_session = distractors_per_session
        self.seed = seed

    def load(self) -> list[Example]:
        rng = random.Random(self.seed)
        examples: list[Example] = []
        for i in range(self.num_examples):
            examples.append(self._make_example(i, rng))
        return examples

    def _fact_sentence(self, entity: str, attr: str, value: str) -> str:
        return f"The {attr} of {entity} is {value}."

    def _make_example(self, idx: int, rng: random.Random) -> Example:
        entity = rng.choice(_ENTITIES)
        attr, values = rng.choice(_ATTRIBUTES)
        value = rng.choice(values)
        is_update = (idx % 4 == 3)  # ~25% knowledge-update questions
        target_session = rng.randrange(self.num_sessions)

        sessions: list[list[Turn]] = []
        for sid in range(self.num_sessions):
            turns: list[Turn] = []
            # Distractor facts about other entities/attributes.
            for _ in range(self.distractors_per_session):
                de = rng.choice([e for e in _ENTITIES if e != entity])
                da, dv = rng.choice(_ATTRIBUTES)
                turns.append(Turn("user", self._fact_sentence(de, da, rng.choice(dv)), sid))
                turns.append(Turn("assistant", "Noted.", sid))
            sessions.append(turns)

        # Inject the target fact into its session.
        sessions[target_session].insert(
            0, Turn("user", self._fact_sentence(entity, attr, value), target_session))
        sessions[target_session].insert(1, Turn("assistant", "Got it, I'll remember that.", target_session))

        final_value = value
        qtype = "single-session-user"
        # The gold memory is the chunk that states the (final) answer value.
        gold_memory_texts = [self._fact_sentence(entity, attr, value)]
        if is_update and target_session < self.num_sessions - 1:
            new_value = rng.choice([v for v in values if v != value] or [value])
            upd_session = rng.randrange(target_session + 1, self.num_sessions)
            upd_text = f"Actually, update it: {self._fact_sentence(entity, attr, new_value)}"
            sessions[upd_session].insert(0, Turn("user", upd_text, upd_session))
            sessions[upd_session].insert(1, Turn("assistant", "Updated.", upd_session))
            final_value = new_value
            qtype = "knowledge-update"
            gold_memory_texts = [upd_text]

        question = f"What is the {attr} of {entity}?"
        return Example(
            example_id=f"syn-{idx:04d}",
            sessions=sessions,
            question=question,
            answer=final_value,
            question_type=qtype,
            n_atomic_facts=1,
            gold_memory_texts=gold_memory_texts,
            metadata={"entity": entity, "attribute": attr, "is_update": is_update},
        )
