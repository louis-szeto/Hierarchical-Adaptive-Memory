"""Command-line interface.

    ham run             --config configs/smoke.yaml --out results/smoke
    ham report          --run-dir results/smoke --out results/smoke/paper_artifacts
    ham export          --run-dir results/smoke --out results/smoke/paper_artifacts   (alias of report)
    ham info                                                                                (env diagnostics)
    ham arch-demo                                                                           (toy HAM-layer demo)
    ham finetune         --config configs/finetune_smoke.yaml --out results/finetune_smoke   (stage-C cost-to-target)
    ham finetune-report  --run-dir results/finetune_smoke --out results/finetune_smoke/artifacts
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import load_config


def _cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.conditions:
        cfg.conditions = args.conditions.split(",")
    if args.limit is not None:
        cfg.dataset.sample_limit = args.limit
        cfg.dataset.num_examples = args.limit
    from .runner import run_experiment

    summary = run_experiment(cfg, args.out)
    print(json.dumps({
        "out_dir": summary["out_dir"],
        "is_smoke": summary["is_smoke"],
        "n_examples": summary["n_examples"],
        "conditions": summary["conditions"],
    }, indent=2))
    if summary["is_smoke"]:
        print("\n[NOTE] Mock backend => SMOKE TEST outputs. Not scientific results.",
              file=sys.stderr)
    return 0


def _cmd_report(args) -> int:
    from .report import generate

    res = generate(args.run_dir, args.out)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _cmd_info(args) -> int:
    from .manifest import build_manifest

    m = build_manifest({}, "-")
    print(json.dumps({"harness_version": __version__,
                      "python": m["python"], "platform": m["platform"],
                      "packages": m["packages"]}, indent=2))
    return 0


def _cmd_arch_demo(args) -> int:
    """Run the architecture-level HAM toy integration (requires torch)."""
    from .architecture import TORCH_AVAILABLE

    if not TORCH_AVAILABLE:
        from .architecture import _INSTALL_HINT

        print(json.dumps({"error": _INSTALL_HINT}), file=sys.stderr)
        return 1
    from .architecture.toy import run_toy_demo

    result = run_toy_demo(block=args.block, fusion=args.fusion)
    print(json.dumps(result, indent=2))
    if not result["invariants_ok"]:
        print("\n[FAIL] architecture invariants not satisfied.", file=sys.stderr)
        return 1
    print("\n[OK] toy HAM layer: shapes preserved, frozen=no-grad, "
          "trainable router/fusion got grads, frozen base did not.", file=sys.stderr)
    return 0


def _cmd_finetune(args) -> int:
    """Run the stage-C fine-tuning cost-to-target experiment."""
    from .config import load_finetune_config
    from .training.runner import run_finetune

    cfg = load_finetune_config(args.config)
    if args.limit is not None:
        cfg.dataset.sample_limit = args.limit
        cfg.dataset.num_examples = args.limit
    summary = run_finetune(cfg, args.out)
    ratio = summary["cost_ratio"]
    print(json.dumps({
        "out_dir": summary["out_dir"], "experiment": summary["experiment"],
        "is_smoke": summary["is_smoke"], "trainer": summary["trainer"],
        "target_stage": summary["target_stage"],
        "base_weights_changed": summary["base_weights_changed"],
        "target_accuracy": summary["target_accuracy"],
        "target_kind": summary["target_kind"],
        "cost_ratio_ham_over_weights": {
            k: v for k, v in ratio.items() if k != "interpretation"},
        "zeroshot_forgetting": summary.get("zeroshot"),
    }, indent=2))
    if summary["is_smoke"]:
        print("\n[NOTE] Mock trainer => SMOKE TEST outputs. Not scientific results.",
              file=sys.stderr)
    return 0


def _cmd_finetune_report(args) -> int:
    """Build finetune tables/figure from a finetune run dir."""
    from .training.report import generate

    res = generate(args.run_dir, args.out)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _cmd_archbench(args) -> int:
    """Run the stage-F toy architecture memory-block experiment."""
    from .config import load_archbench_config
    from .archbench.runner import run_archbench

    cfg = load_archbench_config(args.config)
    summary = run_archbench(cfg, args.out)
    print(json.dumps({
        "out_dir": summary["out_dir"], "experiment": summary["experiment"],
        "is_smoke": summary["is_smoke"], "trainer": summary["trainer"],
        "task": summary["task"], "n_curves": summary["n_curves"],
    }, indent=2))
    if summary["is_smoke"]:
        print("\n[NOTE] Mock trainer => SMOKE TEST outputs. Not scientific results.",
              file=sys.stderr)
    return 0


def _cmd_archbench_report(args) -> int:
    """Build archbench tables/figures from an archbench run dir."""
    from .archbench.report import generate

    res = generate(args.run_dir, args.out)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _cmd_kvbench(args) -> int:
    """Run the stage-D KV-cache-compression experiment (real frozen model)."""
    from .config import load_kvbench_config
    from .kvbench.runner import run_kvbench

    cfg = load_kvbench_config(args.config)
    summary = run_kvbench(cfg, args.out)
    print(json.dumps({
        "out_dir": summary["out_dir"], "experiment": summary["experiment"],
        "is_smoke": summary["is_smoke"], "trainer": summary["trainer"],
        "n_results": summary["n_results"],
    }, indent=2))
    if summary["is_smoke"]:
        print("\n[NOTE] Mock trainer => SMOKE TEST outputs. Not scientific results.",
              file=sys.stderr)
    return 0


def _cmd_kvbench_report(args) -> int:
    """Build kvbench tables/figures from a kvbench run dir."""
    from .kvbench.report import generate

    res = generate(args.run_dir, args.out)
    print(json.dumps(res, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ham", description="HAM memory experiment harness")
    p.add_argument("--version", action="version", version=f"ham {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run an experiment from a YAML config")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--conditions", default=None,
                   help="comma-separated override of conditions")
    r.add_argument("--limit", type=int, default=None, help="cap number of examples")
    r.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="build paper tables/figures from a run dir")
    rep.add_argument("--run-dir", required=True)
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=_cmd_report)

    exp = sub.add_parser("export", help="alias of report (paper-artifacts export)")
    exp.add_argument("--run-dir", required=True)
    exp.add_argument("--out", required=True)
    exp.set_defaults(func=_cmd_report)

    info = sub.add_parser("info", help="print environment + package versions")
    info.set_defaults(func=_cmd_info)

    ad = sub.add_parser("arch-demo",
                        help="run the architecture-level HAM toy integration (needs torch)")
    ad.add_argument("--block", default="transformer",
                    choices=["transformer", "recurrent"])
    ad.add_argument("--fusion", default="cross_attention",
                    choices=["cross_attention", "gated_residual"])
    ad.set_defaults(func=_cmd_arch_demo)

    ft = sub.add_parser("finetune",
                        help="run the stage-C fine-tuning cost-to-target experiment")
    ft.add_argument("--config", required=True)
    ft.add_argument("--out", required=True)
    ft.add_argument("--limit", type=int, default=None, help="cap number of examples")
    ft.set_defaults(func=_cmd_finetune)

    ftr = sub.add_parser("finetune-report",
                         help="build finetune tables/figure from a finetune run dir")
    ftr.add_argument("--run-dir", required=True)
    ftr.add_argument("--out", required=True)
    ftr.set_defaults(func=_cmd_finetune_report)

    ab = sub.add_parser("archbench",
                        help="run the stage-F toy architecture memory-block experiment")
    ab.add_argument("--config", required=True)
    ab.add_argument("--out", required=True)
    ab.set_defaults(func=_cmd_archbench)

    abr = sub.add_parser("archbench-report",
                         help="build archbench tables/figures from an archbench run dir")
    abr.add_argument("--run-dir", required=True)
    abr.add_argument("--out", required=True)
    abr.set_defaults(func=_cmd_archbench_report)

    kv = sub.add_parser("kvbench",
                        help="run the stage-D KV-cache-compression experiment")
    kv.add_argument("--config", required=True)
    kv.add_argument("--out", required=True)
    kv.set_defaults(func=_cmd_kvbench)

    kvr = sub.add_parser("kvbench-report",
                         help="build kvbench tables/figures from a kvbench run dir")
    kvr.add_argument("--run-dir", required=True)
    kvr.add_argument("--out", required=True)
    kvr.set_defaults(func=_cmd_kvbench_report)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
