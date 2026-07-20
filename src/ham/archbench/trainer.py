"""Real torch trainer for the stage-F memory-block experiment.

Trains a :class:`ToyMemoryLM` on a corpus (recall or lm) under one memory-block
condition, recording per-checkpoint quality and byte-honest memory size. The
memory store accumulates across the whole run (FIFO at capacity, consolidated
each checkpoint), so its size is a meaningful quantity to compare across
conditions. Wall-clock/latency is intentionally not recorded (not a universal
metric). Lazy-imports torch and fails loudly with install guidance if absent.

Two regimes share this trainer:
- ``pretrain`` (default): the model starts from random init and is trained from
  scratch. Drift is measured from the random init (training movement).
- ``finetune``: an optional ``init_state_dict`` (e.g. the saved final weights of
  a pretrain run for the same condition) is loaded BEFORE training, so the
  model starts from the pretrained checkpoint and drift is measured from those
  pretrained weights (a real catastrophic-forgetting proxy, not training
  movement from random init). After ``run()``, the trainer also exposes the
  final state_dict via :pyattr:`final_state_dict` so the runner can persist it.
"""

from __future__ import annotations

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
    """Trains one (condition x redundancy x corpus) configuration under one
    regime (``pretrain`` or ``finetune``).

    For ``finetune`` the caller passes ``init_state_dict`` (the loaded pretrained
    weights for this condition); the model loads them before training so the
    L2 weight drift recorded at each checkpoint is ``||w - w_pretrained||_2``,
    not ``||w - w_random||_2``. After ``run()`` the trainer exposes
    ``final_state_dict`` (a detached CPU clone of the trained weights) so the
    runner can persist it to disk for downstream fine-tune runs.
    """

    def __init__(self, cfg: ArchBenchExperimentConfig, condition: str,
                 redundancy: float, corpus: Corpus, device: str, *,
                 init_state_dict: dict | None = None,
                 regime: str = "pretrain"):
        try:
            import torch  # noqa: F401
        except Exception as exc:  # fail loudly
            raise RuntimeError(_INSTALL_HINT) from exc
        if regime not in ("pretrain", "finetune"):
            raise ValueError(
                f"TorchArchTrainer.regime must be 'pretrain'/'finetune', got {regime!r}")
        if regime == "finetune" and init_state_dict is None:
            raise ValueError(
                "finetune regime requires init_state_dict (the pretrained weights)")
        self.cfg = cfg
        self.ab = cfg.archbench
        self.condition = condition
        self.redundancy = redundancy
        self.corpus = corpus
        self.device = _resolve_device(device)
        self.init_state_dict = init_state_dict
        self.regime = regime
        self._final_state_dict: dict | None = None

    @property
    def final_state_dict(self) -> dict | None:
        """Detached CPU state_dict of the trained model (None before ``run()``)."""
        return self._final_state_dict

    def run(self) -> list[ArchCheckpoint]:
        import torch
        torch.manual_seed(self.cfg.seed)
        model = ToyMemoryLM(self.cfg, self.condition).to(self.device)
        # Load the pretrained checkpoint before training (finetune regime).
        # For pretrain, model stays at random init.
        if self.init_state_dict is not None:
            model.load_state_dict(self.init_state_dict)
        store = model.new_store()  # None for no_memory
        # Snapshot the INITIAL parameters right after construction (or after
        # loading the pretrained checkpoint for finetune) so each checkpoint can
        # record the L2 weight drift ||w - w_init||_2 = sqrt(sum((p - p_init)**2)).
        # For pretrain w_init is the random init; for finetune w_init is the
        # loaded pretrained checkpoint, so drift is a catastrophic-forgetting
        # proxy (how far fine-tuning pushed the weights off the pretrained point).
        init_params = [p.detach().clone() for p in model.parameters()]
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

        def _drift_rms() -> float:
            """sqrt(sum((p - p_init)**2)) over all parameters (the L2 norm of
            the weight drift from the init point). For finetune, p_init is the
            pretrained checkpoint, so this is the drift FROM the pretrained
            weights. The naming ``_rms`` mirrors the legacy stage-C field."""
            sq = 0.0
            for p0, p in zip(init_params, model.parameters()):
                d = p.detach().float() - p0.detach().float()
                sq += float((d * d).sum().item())
            return float(sq) ** 0.5

        def _eval(step, tokens, loss):
            model.eval()
            with torch.no_grad():
                logits = model(eval_ids, store, write=False).cpu().numpy()
            quality = quality_metric(logits, eval_tg, eval_qm)
            mem_bytes = store.byte_size() if store is not None else 0
            drift = _drift_rms()
            model.train()
            return ArchCheckpoint(step=step, tokens_seen=tokens,
                                  train_loss=loss, quality=quality,
                                  memory_bytes=mem_bytes,
                                  redundancy=self.redundancy, condition=self.condition,
                                  regime=self.regime, drift_rms=drift)

        model.train()
        tokens_seen = 0
        losses: list[float] = []
        idx = 0
        if 0 in ckpt_set:
            curve.append(_eval(0, 0, None))
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
                avg_loss = (sum(losses) / len(losses)) if losses else None
                curve.append(_eval(step, tokens_seen, avg_loss))
        # Persist the final trained weights so the runner can save them to disk
        # (pretrain) or compare drift endpoints (finetune). CPU + detach so the
        # tensor is independent of the model graph and movable across devices.
        self._final_state_dict = {
            k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        return curve
