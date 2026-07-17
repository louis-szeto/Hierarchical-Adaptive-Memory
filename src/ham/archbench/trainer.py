"""Real torch trainer for the stage-F memory-block experiment.

Trains a :class:`ToyMemoryLM` from scratch on a corpus (recall or lm) under one
memory-block condition, recording per-checkpoint quality, byte-honest memory
size, and per-stream inference latency. The memory store accumulates across the
whole run (FIFO at capacity, consolidated each checkpoint), so its size is a
meaningful quantity to compare across conditions. Lazy-imports torch and fails
loudly with install guidance if absent.
"""

from __future__ import annotations

import time

from ..config import ArchBenchExperimentConfig
from .memory import build_memory_store
from .model import ToyMemoryLM
from .protocol import ArchCheckpoint, checkpoint_steps
from .task import Corpus, quality_metric

_INSTALL_HINT = (
    "the torch archbench trainer requires torch; install with `pip install -e \".[hf]\"`."
)


def _resolve_device(name: str) -> str:
    if name == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return name


class TorchArchTrainer:
    """Trains one (condition x redundancy x corpus) configuration."""

    def __init__(self, cfg: ArchBenchExperimentConfig, condition: str,
                 redundancy: float, corpus: Corpus, device: str):
        try:
            import torch  # noqa: F401
        except Exception as exc:  # fail loudly
            raise RuntimeError(_INSTALL_HINT) from exc
        self.cfg = cfg
        self.ab = cfg.archbench
        self.condition = condition
        self.redundancy = redundancy
        self.corpus = corpus
        self.device = _resolve_device(device)

    def run(self) -> list[ArchCheckpoint]:
        import torch
        torch.manual_seed(self.cfg.seed)
        model = ToyMemoryLM(self.cfg, self.condition).to(self.device)
        store = model.new_store()  # None for no_memory
        opt_cls = torch.optim.AdamW if self.ab.optimizer == "adamw" else torch.optim.SGD
        optimizer = opt_cls(model.parameters(), lr=self.ab.learning_rate)

        ab = self.ab
        V = ab.vocab
        ids = self.corpus.input_ids
        tg = self.corpus.targets
        qm = self.corpus.quality_mask
        n_train = ids.shape[0]
        eval_ids = torch.from_numpy(ids[: min(ab.n_eval_streams, n_train)]).to(self.device)
        eval_tg = tg[: eval_ids.shape[0]]
        eval_qm = qm[: eval_ids.shape[0]]

        ckpt_set = set(checkpoint_steps(ab.max_steps, ab.checkpoint_every))
        curve: list[ArchCheckpoint] = []

        def _eval(step, tokens, wall, loss):
            model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(eval_ids, store, write=False).cpu().numpy()
            latency = (time.perf_counter() - t0) / max(1, eval_ids.shape[0])
            quality = quality_metric(logits, eval_tg, eval_qm)
            mem_bytes = store.byte_size() if store is not None else 0
            model.train()
            return ArchCheckpoint(step=step, tokens_seen=tokens, wall_clock_s=wall,
                                  train_loss=loss, quality=quality,
                                  memory_bytes=mem_bytes, inference_latency_s=latency,
                                  redundancy=self.redundancy, condition=self.condition,
                                  regime="pretrain")

        model.train()
        wall0 = time.perf_counter()
        tokens_seen = 0
        losses: list[float] = []
        idx = 0
        if 0 in ckpt_set:
            curve.append(_eval(0, 0, 0.0, None))
        for step in range(1, ab.max_steps + 1):
            batch_idx = [(idx + k) % n_train for k in range(ab.batch_size)]
            idx = (idx + ab.batch_size) % n_train
            b_ids = torch.from_numpy(ids[batch_idx]).to(self.device)
            b_tg = torch.from_numpy(tg[batch_idx]).to(self.device)
            logits = model(b_ids, store, write=True)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, V), b_tg.reshape(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            tokens_seen += int((b_ids != 0).sum().item())
            if step in ckpt_set:
                if store is not None:
                    store.consolidate()
                wall = time.perf_counter() - wall0
                avg_loss = (sum(losses) / len(losses)) if losses else None
                curve.append(_eval(step, tokens_seen, wall, avg_loss))
        return curve
