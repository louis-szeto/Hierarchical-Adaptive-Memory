"""Paper-ready artifacts for the stage-F archbench experiment, from *actual* run
outputs only.

Headline table = HAM/standard bytes-ratios vs redundancy (the slope
proves 'frequency' is the mechanism). Mock-trainer output is watermarked
``SMOKE TEST``; empty run dirs yield EMPTY TEMPLATE tables and no figures.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from ..report import POC_BANNER, WATERMARK


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def _fmt(v, nd=3):
    if v is None:
        return "n.a."
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def _is_smoke(manifest) -> bool:
    return bool(manifest.get("is_smoke")) if manifest else False


def _poc_banner(manifest):
    if not manifest or manifest.get("is_smoke"):
        return None
    notes = (manifest.get("config", {}) or {}).get("notes", "") or ""
    if "PROOF OF CONCEPT" in notes.upper():
        return f"{POC_BANNER} (toy architecture)"
    return None


def _write_redundancy_table(out_dir, aggregate, smoke) -> str:
    """Headline: bytes/quality ratios of each condition vs standard, per
    redundancy level. The bytes ratio should DROP toward 0 for HAM as redundancy
    rises -- that slope is the proof."""
    path = os.path.join(out_dir, "table_redundancy.md")
    lines = ["# Table AB1 — Memory-block cost vs standard, by redundancy", "",
             "Ratios are condition / standard_memory at the same task & redundancy. "
             "bytes ratio < 1.0 = smaller than standard; quality_delta >~ 0 = iso-quality.",
             "HAM's advantage should GROW with redundancy (the proof that frequency is the mechanism).", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found. Run an archbench experiment first._", ""]
        _write(path, "\n".join(lines))
        return path
    header = "| task | redundancy | condition | bytes_ratio | quality_delta |"
    sep = "|---|---|---|---|---|"
    lines += [header, sep]
    rows = sorted(aggregate.values(),
                  key=lambda e: (e["task"], e["redundancy"], e["condition"]))
    for e in rows:
        lines.append(
            f"| {e['task']} | {e['redundancy']} | {e['condition']} | "
            f"{_fmt(e['bytes_ratio_vs_standard'])} | "
            f"{_fmt(e['quality_delta_vs_standard'])} |")
    lines.append("")
    _write(path, "\n".join(lines))
    return path


def _write_quality_bytes_table(out_dir, aggregate, smoke) -> str:
    path = os.path.join(out_dir, "table_quality_bytes.md")
    lines = ["# Table AB2 — Quality vs byte-honest memory size (Pareto)", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found._", ""]
        _write(path, "\n".join(lines))
        return path
    header = "| task | redundancy | condition | quality_final | memory_bytes_peak |"
    sep = "|---|---|---|---|---|"
    lines += [header, sep]
    for e in sorted(aggregate.values(), key=lambda e: (e["task"], e["redundancy"], e["condition"])):
        lines.append(f"| {e['task']} | {e['redundancy']} | {e['condition']} | "
                     f"{_fmt(e['quality_final'])} | {e['memory_bytes_peak']} |")
    lines.append("")
    _write(path, "\n".join(lines))
    return path


def _write_finetune_posthoc_table(out_dir, finetune_posthoc, smoke) -> str:
    """Fine-tuning post-hoc on the toy models: standard_memory vs ham_memory
    cost-to-target + L2 weight drift at the parity target. The toy LM is
    trained from scratch (no zero-shot forgetting arm); the diagnostic is the
    drift overhead HAM's extra parameters add to reach the same target."""
    path = os.path.join(out_dir, "table_finetune_posthoc.md")
    delta = (finetune_posthoc or {}).get("noninferiority_delta")
    lines = [
        "# Table AB3 — Fine-tuning post-hoc on the toy models (standard vs HAM memory block)", "",
        "Both arms are toy LMs WITH a memory block, trained from scratch under "
        "the identical config; only the memory-block policy differs "
        "(`standard_memory` = FlatMemory vs `ham_memory` = HamMemory).",
        f"Parity target = max(standard_quality) − δ (δ = {_fmt(delta)}). "
        "Cost = first checkpoint at-or-above the target (no interpolation). "
        "drift = ‖Δw‖₂ = sqrt(sum((p − p_init)²)) over all params at that "
        "checkpoint. ratios < 1.0 = HAM cheaper/smaller-drift; > 1.0 = HAM "
        "more expensive.",
        "The toy is trained from scratch (no pretrained knowledge to forget -> "
        "no zero-shot forgetting arm; the diagnostic is the drift overhead).",
        "",
    ]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    cells = (finetune_posthoc or {}).get("cells") or {}
    if not cells:
        lines += ["_EMPTY TEMPLATE — no standard_memory/ham_memory pair found in the run._", ""]
        _write(path, "\n".join(lines))
        return path
    header = ("| task | redundancy | target | arm | reached | quality@target | "
              "steps | tokens | drift | step ratio | token ratio | drift ratio |")
    sep = "|" + "---|" * 12
    lines += [header, sep]
    for key in sorted(cells.keys()):
        cell = cells[key]
        for arm in ("standard", "ham"):
            a = cell[arm]
            ratio_step = cell["cost_ratio_steps_ham_over_standard"] if arm == "ham" else None
            ratio_tok = cell["cost_ratio_tokens_ham_over_standard"] if arm == "ham" else None
            ratio_drift = cell["drift_ratio_ham_over_standard"] if arm == "ham" else None
            ref = "1 (ref)" if arm == "standard" else None
            lines.append(
                f"| {cell['task']} | {cell['redundancy']} | "
                f"{_fmt(cell['target_quality'])} | {arm} | "
                f"{'yes' if a['reached'] else 'no'} | {_fmt(a['quality_at_target'])} | "
                f"{_fmt(a['optimizer_steps_to_target'])} | "
                f"{_fmt(a['training_tokens_to_target'])} | "
                f"{_fmt(a['drift_rms_at_target'])} | "
                f"{_fmt(ref) if ref is not None else _fmt(ratio_step)} | "
                f"{_fmt(ref) if ref is not None else _fmt(ratio_tok)} | "
                f"{_fmt(ref) if ref is not None else _fmt(ratio_drift)} |")
    lines.append("")
    _write(path, "\n".join(lines))
    # CSV companion (one row per cell x arm).
    csv_path = os.path.join(out_dir, "table_finetune_posthoc.csv")
    with open(csv_path, "w") as fh:
        fh.write("task,redundancy,target_quality,arm,reached,quality_at_target,"
                 "optimizer_steps_to_target,training_tokens_to_target,"
                 "drift_rms_at_target,cost_ratio_steps_ham_over_standard,"
                 "cost_ratio_tokens_ham_over_standard,"
                 "drift_ratio_ham_over_standard\n")
        for key in sorted(cells.keys()):
            cell = cells[key]
            for arm in ("standard", "ham"):
                a = cell[arm]
                fh.write(",".join(str(v) for v in [
                    cell["task"], cell["redundancy"], cell["target_quality"],
                    arm, a["reached"], a["quality_at_target"],
                    a["optimizer_steps_to_target"], a["training_tokens_to_target"],
                    a["drift_rms_at_target"],
                    cell["cost_ratio_steps_ham_over_standard"] if arm == "ham" else "",
                    cell["cost_ratio_tokens_ham_over_standard"] if arm == "ham" else "",
                    cell["drift_ratio_ham_over_standard"] if arm == "ham" else "",
                ]) + "\n")
    return path


def _figures(out_dir, aggregate, smoke) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _write(os.path.join(out_dir, "FIGURES_SKIPPED.txt"),
               "matplotlib not installed; install the [plot] extra to render figures.\n")
        return []
    if not aggregate:
        return []
    made: list[str] = []

    def _wm(ax):
        if smoke:
            ax.text(0.5, 0.5, "SMOKE TEST", transform=ax.transAxes, fontsize=30, color="red",
                    alpha=0.20, ha="center", va="center", rotation=30, fontweight="bold")

    # AB-F1: bytes_ratio vs redundancy, per condition (the headline).
    by_cond: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in aggregate.values():
        if e["bytes_ratio_vs_standard"] is not None:
            by_cond[e["condition"]].append(
                (e["redundancy"], e["bytes_ratio_vs_standard"]))
    if by_cond:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cond, pts in by_cond.items():
            pts.sort()
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], marker="o", label=f"{cond} (bytes)")
        ax.axhline(1.0, ls=":", color="gray", lw=1)
        ax.set_xlabel("Corpus redundancy (0 = uniform, 1 = highly redundant)")
        ax.set_ylabel("memory bytes / standard_memory")
        ax.set_title("HAM byte advantage grows with redundancy" + ("  [SMOKE TEST]" if smoke else ""))
        ax.legend(fontsize=7)
        _wm(ax)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_advantage_vs_redundancy.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)

    # AB-F2: quality vs bytes Pareto (final checkpoint, averaged over redundancy).
    by_cond_qb: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for e in aggregate.values():
        by_cond_qb[e["condition"]].append((e["memory_bytes_peak"], e["quality_final"]))
    if by_cond_qb:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for cond, pts in by_cond_qb.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=50)
            for x, y in zip(xs, ys):
                ax.annotate(cond, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Peak memory bytes (byte-honest)")
        ax.set_ylabel("Quality (final)")
        ax.set_title("Quality vs memory-bytes Pareto" + ("  [SMOKE TEST]" if smoke else ""))
        _wm(ax)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_quality_vs_bytes.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)
    return made


def generate(run_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    aggregate = _load_json(os.path.join(run_dir, "aggregate.json"))
    manifest = _load_json(os.path.join(run_dir, "manifest.json"))
    _load_jsonl(os.path.join(run_dir, "curve.jsonl"))  # presence check
    smoke = _is_smoke(manifest)

    # The fine-tuning post-hoc block is serialized as a nested key inside
    # aggregate.json by the runner.
    finetune_posthoc = (aggregate or {}).pop("finetune_posthoc", None)

    made = [_write_redundancy_table(out_dir, aggregate, smoke),
            _write_quality_bytes_table(out_dir, aggregate, smoke),
            _write_finetune_posthoc_table(out_dir, finetune_posthoc, smoke)]
    made.extend(_figures(out_dir, aggregate, smoke))

    banner = _poc_banner(manifest)
    label = WATERMARK if smoke else (banner or "measured toy-architecture run")
    _write(os.path.join(out_dir, "RUN_LABEL.txt"), label + "\n")

    index = ["# Archbench (stage F) artifacts", ""]
    if smoke:
        index += [f"> **{WATERMARK}**", ""]
    elif banner:
        index += [f"> **{banner}**", ""]
    if not aggregate:
        index += ["_No run data found; tables written as EMPTY TEMPLATES; no figures created._", ""]
    for m in made:
        index.append(f"- `{os.path.basename(m)}`")
    _write(os.path.join(out_dir, "README.md"), "\n".join(index) + "\n")

    return {"out_dir": out_dir, "is_smoke": smoke, "artifacts": made,
            "had_data": bool(aggregate)}
