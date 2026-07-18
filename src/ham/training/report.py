"""Paper-ready artifacts for the stage-C fine-tuning experiment, from *actual*
run outputs only.

Reads a finetune run dir (curve.jsonl / aggregate.json / stats.json /
manifest.json) and emits Markdown + CSV tables and the accuracy-vs-tokens figure.
With no run data, generators write clearly-labeled EMPTY TEMPLATE files and
refuse to invent values. Mock-trainer output is watermarked ``SMOKE TEST``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from ..report import POC_BANNER, WATERMARK


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_jsonl(path: str):
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
        model = (manifest.get("config", {}) or {}).get("backend", {}).get("model_id", "?")
        return f"{POC_BANNER} (model: {model})"
    return None


_COST_COLUMNS = [
    ("final_accuracy", "FinalAcc"),
    ("max_accuracy", "MaxAcc"),
    ("target_accuracy", "Target"),
    ("reached", "Reached"),
    ("optimizer_steps_to_target", "StepsToTarget"),
    ("training_tokens_to_target", "TokensToTarget"),
    ("drift_rms_at_target", "DriftAtTarget"),
    ("cost_ratio_tokens", "RatioTokens"),
]


def _write_cost_table(out_dir, aggregate, target, smoke) -> str:
    path = os.path.join(out_dir, "table_cost.md")
    lines = ["# Table FT1 — Cost-to-target by leg", "",
             "Training tokens / optimizer-steps / RMS weight drift at the target "
             "knowledge accuracy. RatioTokens = ham_augmented / weights_only "
             "(<1.0 means HAM reached the target cheaper).", "",
             "NEW DESIGN: Two legs trained independently from identical baseline: "
             "weights_only (no-context SFT) vs ham_augmented (context-augmented SFT)."]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    header = "| Arm | " + " | ".join(c[1] for c in _COST_COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COST_COLUMNS) + 1)
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found. Run a finetune experiment first._", "",
                  header, sep, "| _(no data)_ | " + " | ".join("n.a." for _ in _COST_COLUMNS) + " |"]
        _write(path, "\n".join(lines))
        return path
    lines += [f"Target accuracy = {_fmt(target)}", "", header, sep]
    for leg, entry in aggregate.items():
        cells = [_fmt(entry.get(col)) for col, _ in _COST_COLUMNS]
        # weights_only is the reference (ratio 1.00).
        ratio = "1.00 (ref)" if leg == "weights_only" else _fmt(entry.get("cost_ratio_tokens"))
        cells[-1] = ratio
        lines.append(f"| {leg} | " + " | ".join(cells) + " |")
    lines.append("")
    _write(path, "\n".join(lines))
    csv_path = os.path.join(out_dir, "table_cost.csv")
    with open(csv_path, "w") as fh:
        fh.write("leg," + ",".join(c[0] for c in _COST_COLUMNS) + "\n")
        for leg, entry in aggregate.items():
            fh.write(leg + "," + ",".join(str(entry.get(c[0], "")) for c in _COST_COLUMNS) + "\n")
    return path


def _write_curve_table(out_dir, curve_rows, smoke) -> str:
    path = os.path.join(out_dir, "table_curve.md")
    lines = ["# Table FT2 — Accuracy curve (per checkpoint x leg)", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    header = "| Step | TokensSeen | Arm | Accuracy | PromptTok |"
    sep = "|---|---|---|---|---|"
    if not curve_rows:
        lines += ["_EMPTY TEMPLATE — no run data found._", "", header, sep,
                  "| _(no data)_ | n.a. | n.a. | n.a. | n.a. |"]
        _write(path, "\n".join(lines))
        return path
    lines += [header, sep]
    # Aggregate curve.jsonl rows to (step, leg) mean accuracy.
    bucket: dict[tuple, list] = defaultdict(list)
    for r in curve_rows:
        bucket[(r["step"], r["tokens_seen"], r["leg"])].append(r)
    for (step, tokens, leg), rows in sorted(bucket.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        acc = sum(r["correct"] for r in rows) / len(rows)
        ptok = sum(r["prompt_tokens"] for r in rows) / len(rows)
        lines.append(f"| {step} | {tokens} | {leg} | {_fmt(acc)} | {_fmt(ptok)} |")
    lines.append("")
    _write(path, "\n".join(lines))
    return path


def _write_stats_table(out_dir, stats_data, smoke) -> str:
    path = os.path.join(out_dir, "table_stats.md")
    lines = ["# Statistical comparisons (paired, ham_augmented vs weights_only)", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not stats_data or not stats_data.get("comparisons"):
        lines += ["_EMPTY TEMPLATE — no comparisons available._", ""]
        _write(path, "\n".join(lines))
        return path
    lines += [f"Non-inferiority margin δ = {stats_data.get('noninferiority_delta', 'n.a.')}", "",
              "| Checkpoint | Δ correctness | 95% CI | perm p | McNemar p | non-inferior |",
              "|---|---|---|---|---|---|"]
    for name, comp in stats_data["comparisons"].items():
        diff = comp.get("paired_bootstrap_diff", {})
        perm = comp.get("paired_permutation", {})
        mc = comp.get("mcnemar", {})
        ni = comp.get("noninferiority", {})
        ci = f"[{_fmt(diff.get('lo'))}, {_fmt(diff.get('hi'))}]"
        lines.append(
            f"| {name} (step {comp.get('step')}) | {_fmt(diff.get('mean_diff'))} | {ci} | "
            f"{_fmt(perm.get('p_value'))} | {_fmt(mc.get('p_value'))} | {ni.get('non_inferior', 'n.a.')} |")
    lines.append("")
    _write(path, "\n".join(lines))
    return path


def _figure(out_dir, curve_rows, target, smoke) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _write(os.path.join(out_dir, "FIGURES_SKIPPED.txt"),
               "matplotlib not installed; install the [plot] extra to render figures.\n")
        return []
    if not curve_rows:
        return []
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    bucket: dict[tuple, list] = defaultdict(list)
    for r in curve_rows:
        bucket[(r["step"], r["tokens_seen"], r["leg"])].append(r)
    for (step, tokens, leg), rows in bucket.items():
        acc = sum(r["correct"] for r in rows) / len(rows)
        series[leg].append((tokens, acc))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for leg, pts in series.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=leg)
    if target is not None:
        ax.axhline(target, ls="--", color="gray", lw=1, label=f"target={target:g}")
    ax.set_xlabel("Training tokens seen")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs training tokens" + ("  [SMOKE TEST]" if smoke else ""))
    if smoke:
        ax.text(0.5, 0.5, "SMOKE TEST", transform=ax.transAxes, fontsize=34, color="red",
                alpha=0.20, ha="center", va="center", rotation=30, fontweight="bold")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_accuracy_vs_tokens.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return [p]


def generate(run_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    aggregate = _load_json(os.path.join(run_dir, "aggregate.json"))
    stats_data = _load_json(os.path.join(run_dir, "stats.json"))
    manifest = _load_json(os.path.join(run_dir, "manifest.json"))
    curve_rows = _load_jsonl(os.path.join(run_dir, "curve.jsonl"))
    smoke = _is_smoke(manifest)
    target = (manifest or {}).get("target_accuracy")

    made = [_write_cost_table(out_dir, aggregate, target, smoke),
            _write_curve_table(out_dir, curve_rows, smoke),
            _write_stats_table(out_dir, stats_data, smoke)]
    made.extend(_figure(out_dir, curve_rows, target, smoke))

    banner = _poc_banner(manifest)
    label = WATERMARK if smoke else (banner or "measured finetune run")
    _write(os.path.join(out_dir, "RUN_LABEL.txt"), label + "\n")

    index = ["# Fine-tuning experiment artifacts", ""]
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
