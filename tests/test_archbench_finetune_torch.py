"""Torch-only tests for the archbench fine-tune regime (pretrain -> save ->
load -> finetune). Skipped when torch is not installed; the no-torch CI path
exercises the mock finetune regime in ``test_archbench_runner.py``."""

import os

import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from ham.archbench import build_trainer
from ham.archbench.task import build_corpus
from ham.archbench.trainer import TorchArchTrainer
from ham.config import archbench_from_dict


def _cfg(**over):
    base = {"archbench": {
        "trainer": "torch", "task": "recall", "regime": "both",
        "redundancy_levels": [0.0],
        "conditions": ["standard_memory", "ham_memory"],
        "dim": 16, "n_layers": 1, "vocab": 32, "n_heads": 2,
        "memory_layer": 0, "top_k": 2, "capacity": 32,
        "max_steps": 8, "checkpoint_every": 4,
        "seq_len": 8, "n_train_streams": 8, "n_eval_streams": 4,
        "device": "cpu"}}
    base["archbench"].update(over)
    return archbench_from_dict(base)


def _corpus(cfg, seed):
    return build_corpus("recall", n_streams=cfg.archbench.n_train_streams,
                        seq_len=cfg.archbench.seq_len, vocab=cfg.archbench.vocab,
                        n_items=8, redundancy=0.0, seed=seed)


def test_torch_trainer_pretrain_tags_checkpoints_and_exposes_final_state():
    cfg = _cfg()
    corpus = _corpus(cfg, cfg.seed)
    trainer = TorchArchTrainer(cfg, "standard_memory", 0.0, corpus, "cpu",
                               regime="pretrain")
    curve = trainer.run()
    assert all(c.regime == "pretrain" for c in curve)
    assert all(c.condition == "standard_memory" for c in curve)
    sd = trainer.final_state_dict
    assert sd is not None
    assert isinstance(sd, dict)
    # Sanity: the state_dict has the expected toy-LM keys.
    assert any("embed" in k for k in sd.keys())


def test_torch_trainer_finetune_loads_init_state_dict():
    cfg = _cfg()
    # Pretrain: train from random init, capture final state_dict.
    pre_corpus = _corpus(cfg, cfg.seed)
    pre = TorchArchTrainer(cfg, "standard_memory", 0.0, pre_corpus, "cpu",
                           regime="pretrain")
    pre_curve = pre.run()
    pre_sd = pre.final_state_dict
    assert pre_sd is not None

    # Finetune: train from the pretrained checkpoint, capture drift at step 0.
    fin_corpus = _corpus(cfg, cfg.seed + 1001)
    fin = TorchArchTrainer(cfg, "standard_memory", 0.0, fin_corpus, "cpu",
                           init_state_dict=pre_sd, regime="finetune")
    fin_curve = fin.run()
    assert all(c.regime == "finetune" for c in fin_curve)
    # Step-0 drift is 0 (model loads the pretrained init exactly).
    assert fin_curve[0].step == 0
    assert abs(fin_curve[0].drift_rms) < 1e-6
    # After training, drift is > 0 (fine-tuning moved off the pretrained point).
    assert fin_curve[-1].drift_rms > 0.0


def test_torch_trainer_finetune_without_init_state_raises():
    cfg = _cfg()
    corpus = _corpus(cfg, cfg.seed)
    with pytest.raises(ValueError, match="init_state_dict"):
        TorchArchTrainer(cfg, "standard_memory", 0.0, corpus, "cpu",
                         regime="finetune")


def test_torch_trainer_bad_regime_raises():
    cfg = _cfg()
    corpus = _corpus(cfg, cfg.seed)
    with pytest.raises(ValueError, match="regime"):
        TorchArchTrainer(cfg, "standard_memory", 0.0, corpus, "cpu",
                         regime="bogus")


def test_torch_finetune_drift_from_pretrained_not_from_random():
    # The drift measured in finetune must be from the pretrained weights, NOT
    # from random init. We verify by constructing a trainer whose init is the
    # pretrained checkpoint: at step 0 the model equals the pretrained point,
    # so drift must be ~0. (If drift were from random init it would be > 0.)
    cfg = _cfg()
    pre_corpus = _corpus(cfg, cfg.seed)
    pre = TorchArchTrainer(cfg, "ham_memory", 0.0, pre_corpus, "cpu",
                           regime="pretrain")
    pre.run()
    pre_sd = pre.final_state_dict

    fin_corpus = _corpus(cfg, cfg.seed + 999)
    fin = TorchArchTrainer(cfg, "ham_memory", 0.0, fin_corpus, "cpu",
                           init_state_dict=pre_sd, regime="finetune")
    curve = fin.run()
    assert curve[0].drift_rms < 1e-6
    # After one checkpoint of training, the finetune has moved off pre_sd.
    assert curve[-1].drift_rms > curve[0].drift_rms


def test_build_trainer_forwards_regime_and_init_state():
    cfg = _cfg()
    pre_corpus = _corpus(cfg, cfg.seed)
    pre = build_trainer(cfg, "standard_memory", 0.0, pre_corpus, "cpu",
                        regime="pretrain")
    pre.run()
    pre_sd = pre.final_state_dict

    fin_corpus = _corpus(cfg, cfg.seed + 1001)
    fin = build_trainer(cfg, "standard_memory", 0.0, fin_corpus, "cpu",
                        regime="finetune", init_state_dict=pre_sd)
    curve = fin.run()
    assert all(c.regime == "finetune" for c in curve)


def test_pretrain_state_dicts_saved_to_disk_and_reloadable(tmp_path):
    # End-to-end: a regime=both run saves <out>/pretrained_checkpoints/<cond>.pt
    # files that torch can reload, so a separate finetune-only invocation can
    # read them via finetune_init_from_dir.
    from ham.archbench.runner import run_archbench, _load_init_state_dict

    out = str(tmp_path / "run")
    cfg = _cfg(regime="both")
    run_archbench(cfg, out)

    ckpt_dir = os.path.join(out, "pretrained_checkpoints")
    assert os.path.exists(os.path.join(ckpt_dir, "standard_memory.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "ham_memory.pt"))

    sd = _load_init_state_dict(out, "standard_memory")
    assert isinstance(sd, dict)
    # The reloaded state_dict can initialize a fresh toy LM (round-trip).
    cfg2 = _cfg(regime="finetune")
    heldout = _corpus(cfg2, cfg2.seed + 1001)
    fin = TorchArchTrainer(cfg2, "standard_memory", 0.0, heldout, "cpu",
                           init_state_dict=sd, regime="finetune")
    curve = fin.run()
    assert curve[0].drift_rms < 1e-6   # loaded the pretrained init exactly
