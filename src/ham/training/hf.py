"""Real Hugging Face SFT trainer (torch required).

NEW DESIGN: trains ONE leg with a given prompt mode.

- ``weights_only`` leg:     SFT on no-context prompts (``Question -> Answer``)
- ``ham_augmented`` leg:    SFT on context-augmented prompts (``Context + Question -> Answer``)

The HAM store is built once from the fact corpus and reused for all context-augmented
training examples. Each leg has its own optimizer and trains from the same baseline.
"""

from __future__ import annotations

import time

from ..backends.hf import HFBackend
from ..config import FinetuneExperimentConfig
from ..datasets.base import Example
from . import eval as eval_mod
from .protocol import CheckpointEval, LegTrainer, checkpoint_steps, LEG_TO_PROMPT_MODE

_INSTALL_HINT = (
    "the hf trainer requires torch + transformers; install with "
    "`pip install -e \".[hf]\"`."
)


def _sft_sequences(cfg: FinetuneExperimentConfig, leg: str,
                   examples: list[Example], corpus_facts: list[str]) -> list[tuple[str, str]]:
    """Return (prompt, answer) SFT pairs for ONE leg.

    The prompt matches the leg's eval format so the model learns to produce answers
    in the exact format it will be evaluated on.

    - ``weights_only``:     (NO_CONTEXT_TEMPLATE.format(question=q), answer)
    - ``ham_augmented``:    (CONTEXT_TEMPLATE.format(context=retrieved, question=q), answer)

    The HAM store is built fresh to retrieve context for each training example.
    """
    from .eval import NO_CONTEXT_TEMPLATE, CONTEXT_TEMPLATE, build_leg_memory
    from ..embeddings import build_embedder

    # Build a temporary embedder for context retrieval (same cfg)
    embedder = build_embedder(cfg.embedding)
    mem = build_leg_memory(cfg, leg, embedder, corpus_facts)
    prompt_mode = LEG_TO_PROMPT_MODE[leg]

    pairs = []
    for ex in examples:
        if prompt_mode == "no_context":
            prompt = NO_CONTEXT_TEMPLATE.format(question=ex.question)
        else:  # context_augmented
            context, _ = mem.build_context(ex.question)
            prompt = CONTEXT_TEMPLATE.format(context=context, question=ex.question)
        pairs.append((prompt, ex.answer))
    return pairs


class HFLegTrainer:
    """Real SFT loop for ONE leg off the HF backend's model/tokenizer."""

    def __init__(self, leg: str, backend: HFBackend, embedder, cfg: FinetuneExperimentConfig,
                 examples: list[Example], corpus_facts: list[str]):
        if leg not in ("weights_only", "ham_augmented"):
            raise ValueError(f"leg must be 'weights_only' or 'ham_augmented', got {leg!r}")
        if not isinstance(backend, HFBackend):
            raise RuntimeError("the hf trainer requires backend.kind == 'hf'")
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        self.leg = leg
        self.backend = backend
        self.embedder = embedder
        self.cfg = cfg
        self.examples = examples
        self.corpus_facts = corpus_facts

    def run(self) -> list[CheckpointEval]:
        torch = self.backend._torch
        ft = self.cfg.finetune
        tok = self.backend.tokenizer
        model = self.backend.model
        device = self.backend.device
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or "<pad>"

        # Build SFT sequences for this leg
        seqs = _sft_sequences(self.cfg, self.leg, self.examples, self.corpus_facts)
        if not seqs:
            raise RuntimeError("no SFT sequences: provide a non-empty fact corpus")

        # Tokenize with prompt masking (loss only on answer tokens)
        pad_id = tok.pad_token_id
        eos_id = tok.eos_token_id
        ex_ids: list[list[int]] = []
        ex_labels: list[list[int]] = []
        for prompt, answer in seqs:
            p_ids = tok(prompt, add_special_tokens=False, truncation=True,
                        max_length=256)["input_ids"]
            full = prompt + " " + answer
            f_ids = tok(full, add_special_tokens=False, truncation=True,
                        max_length=384)["input_ids"]
            if eos_id is not None:
                f_ids = f_ids + [eos_id]
            labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[min(len(p_ids), len(f_ids)):]
            ex_ids.append(f_ids)
            ex_labels.append(labels)

        maxlen = max(len(x) for x in ex_ids)
        input_ids = torch.full((len(ex_ids), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ex_ids), maxlen), dtype=torch.long)
        labels_all = torch.full((len(ex_ids), maxlen), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(ex_ids, ex_labels)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, :len(ids)] = 1
            labels_all[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
        n = input_ids.shape[0]

        torch.manual_seed(self.cfg.seed)
        opt_cls = torch.optim.AdamW if ft.optimizer == "adamw" else torch.optim.SGD
        optimizer = opt_cls(model.parameters(), lr=ft.learning_rate)

        ckpt_steps = checkpoint_steps(ft.max_steps, ft.checkpoint_every)
        ckpt_set = set(ckpt_steps)
        curve: list[CheckpointEval] = []

        def _eval(step: int, tokens: int, wall: float, loss,
                  force_no_context: bool = False) -> CheckpointEval:
            model.eval()
            results = eval_mod.eval_leg(
                self.backend, self.cfg, self.leg, self.embedder,
                self.corpus_facts, self.examples,
                force_no_context=force_no_context)
            model.train()
            return CheckpointEval(step=step, tokens_seen=tokens, wall_clock_s=wall,
                                  train_loss=loss, leg=self.leg, results=results)

        model.train()
        wall0 = time.perf_counter()
        tokens_seen = 0
        losses: list[float] = []
        idx = 0
        if 0 in ckpt_set:
            # Step 0 = brand-new model: evaluate BOTH legs no-context so they
            # share an identical 0-accuracy baseline (no retrieval echo inflation).
            curve.append(_eval(0, 0, 0.0, None, force_no_context=True))
        for step in range(1, ft.max_steps + 1):
            batch_idx = [(idx + k) % n for k in range(ft.batch_size)]
            idx = (idx + ft.batch_size) % n
            b_ids = input_ids[batch_idx].to(device)
            b_attn = attn[batch_idx].to(device)
            labels = labels_all[batch_idx].to(device)
            optimizer.zero_grad()
            out = model(input_ids=b_ids, attention_mask=b_attn, labels=labels)
            out.loss.backward()
            optimizer.step()
            losses.append(float(out.loss.item()))
            # Count TRAINING tokens as the supervised target tokens (the answers
            # the loss is computed on), NOT the full input sequence. Both legs
            # fine-tune on the identical Q->A data, so this is equal across legs
            # by construction; the ham leg's retrieved context is the HAM memory
            # system (a read cost, captured in wall-clock / prompt tokens), not
            # training data.
            tokens_seen += int((labels >= 0).sum().item())
            if step in ckpt_set:
                wall = time.perf_counter() - wall0
                avg_loss = (sum(losses) / len(losses)) if losses else None
                curve.append(_eval(step, tokens_seen, wall, avg_loss))
        model.eval()
        return curve
