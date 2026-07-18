"""Paper-ready artifact generation from *actual* run outputs only.

Reads a run directory (aggregate.json / per_example.jsonl / stats.json / manifest.json)
and emits Markdown + CSV tables and figures. If no real run data exists, the
generators write clearly-labeled EMPTY TEMPLATE files and refuse to invent values.
Every figure built from mock-backend data is watermarked "SMOKE TEST".
"""

from __future__ import annotations

import json
import os

WATERMARK = "SMOKE TEST — mock backend, NOT scientific results"
POC_BANNER = "REAL MODEL, PROOF OF CONCEPT — small model / tiny sample, NOT publication evidence"


def _poc_banner(manifest) -> str | None:
    """Return a PoC banner for a real (non-smoke) run whose config notes flag it
    as a proof of concept; otherwise None."""
    if not manifest:
        return None
    if manifest.get("is_smoke"):
        return None
    notes = (manifest.get("config", {}) or {}).get("notes", "") or ""
    if "PROOF OF CONCEPT" in notes.upper():
        model = (manifest.get("config", {}) or {}).get("backend", {}).get("model_id", "?")
        return f"{POC_BANNER} (model: {model})"
    return None


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


def _is_smoke(manifest, aggregate) -> bool:
    if manifest and "is_smoke" in manifest:
        return bool(manifest["is_smoke"])
    if aggregate:
        return any(e.get("is_smoke") for e in aggregate.values())
    return False


def _fmt(v, nd=3):
    if v is None:
        return "n.a."
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


# --- tables -----------------------------------------------------------------

_MAIN_COLUMNS = [
    ("task_score_mean", "TaskScore"),
    ("exact_match_mean", "EM"),
    ("f1_mean", "F1"),
    ("prompt_tokens_mean", "PromptTok"),
    ("total_tokens_mean", "TotalTok"),
    ("physical_serialized_bytes_mean", "PhysBytes"),
    ("logical_memory_bytes_mean", "LogicalBytes"),
    ("bytes_per_fact_mean", "Bytes/Fact"),
    ("compression_ratio_mean", "CompRatio"),
    ("index_size_bytes_mean", "IndexBytes"),
]


def _write_main_table(out_dir, aggregate, smoke) -> str:
    path = os.path.join(out_dir, "table_main.md")
    lines = ["# Table T1 — Main results", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found. Run an experiment first._", ""]
        header = "| Condition | " + " | ".join(c[1] for c in _MAIN_COLUMNS) + " |"
        sep = "|" + "---|" * (len(_MAIN_COLUMNS) + 1)
        lines += [header, sep, "| _(no data)_ | " + " | ".join("n.a." for _ in _MAIN_COLUMNS) + " |"]
        _write(path, "\n".join(lines))
        return path
    header = "| Condition | " + " | ".join(c[1] for c in _MAIN_COLUMNS) + " |"
    sep = "|" + "---|" * (len(_MAIN_COLUMNS) + 1)
    lines += [header, sep]
    for cond, entry in aggregate.items():
        cells = [_fmt(entry.get(col)) for col, _ in _MAIN_COLUMNS]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines.append("")
    _write(path, "\n".join(lines))
    # CSV mirror.
    csv_path = os.path.join(out_dir, "table_main.csv")
    with open(csv_path, "w") as fh:
        fh.write("condition," + ",".join(c[0] for c in _MAIN_COLUMNS) + "\n")
        for cond, entry in aggregate.items():
            fh.write(cond + "," + ",".join(str(entry.get(col, "")) for col, _ in _MAIN_COLUMNS) + "\n")
    return path


# Baselines-only, target-stage/method/outcome table (research addendum §2/§7).
# Executable conditions only; NEVER mixes external papers' reported numbers.
_BASELINE_TABLE_ORDER = [
    "memory_off", "full_history", "uncompressed_rag", "recency_fifo",
    "static_prototype", "uniform_quantization", "ham_memory",
]
_BASELINE_COLUMNS = [
    ("integration_mode", "Integration", False),
    ("base_weights_changed", "BaseWtsΔ", False),
    ("persistent", "Persistent", False),
    ("consolidation", "Consolid.", False),
    ("adaptive_precision", "AdaptPrec", False),
    ("task_score_mean", "TaskScore", True),
    ("retrieval_recall_at_k_mean", "Recall@k", True),
    ("retrieval_mrr_mean", "MRR", True),
    ("prompt_tokens_mean", "PromptTok", True),
    ("physical_serialized_bytes_mean", "PhysBytes", True),
    ("compression_ratio_mean", "CompRatio", True),
    ("peak_cpu_rss_bytes_mean", "PeakRSS", True),
]


def _write_baseline_table(out_dir, aggregate, smoke) -> str:
    path = os.path.join(out_dir, "table_baselines.md")
    lines = ["# Table T2 — Executable baselines (target stage / method / outcome)", "",
             "_Measured local-run values only. Every condition is an implemented "
             "analogue under this harness, not a reproduction of any external paper; "
             "no external reported metrics appear here._", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    header = "| Condition | " + " | ".join(c[1] for c in _BASELINE_COLUMNS) + " |"
    sep = "|" + "---|" * (len(_BASELINE_COLUMNS) + 1)
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found. Run an experiment first._", "",
                  header, sep,
                  "| _(no data)_ | " + " | ".join("n.a." for _ in _BASELINE_COLUMNS) + " |"]
        _write(path, "\n".join(lines))
        return path
    lines += [header, sep]
    present = [c for c in _BASELINE_TABLE_ORDER if c in aggregate]
    for cond in present:
        entry = aggregate[cond]
        cells = [_fmt(entry.get(col)) for col, _, _ in _BASELINE_COLUMNS]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines.append("")
    _write(path, "\n".join(lines))
    csv_path = os.path.join(out_dir, "table_baselines.csv")
    with open(csv_path, "w") as fh:
        fh.write("condition," + ",".join(c[0] for c in _BASELINE_COLUMNS) + "\n")
        for cond in present:
            entry = aggregate[cond]
            fh.write(cond + "," + ",".join(str(entry.get(col, "")) for col, _, _ in _BASELINE_COLUMNS) + "\n")
    return path


_DELTA_COLS = [
    ("task_score", "ΔTaskScore"),
    ("retrieval_recall_at_k", "ΔRecall@k"),
    ("retrieval_mrr", "ΔMRR"),
    ("prompt_tokens", "ΔPromptTok"),
    ("physical_serialized_bytes", "ΔPhysBytes"),
]


def _write_deltas_table(out_dir, aggregate, smoke) -> str:
    path = os.path.join(out_dir, "table_deltas.md")
    lines = ["# Table T3 — Per-condition deltas vs anchor baselines", "",
             "Δ against `memory_off` (no persistent memory) and `uncompressed_rag` "
             "(uncompressed retrieval). Positive = condition minus anchor.", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found._", ""]
        _write(path, "\n".join(lines))
        return path
    for label in ("vs_memory_off", "vs_uncompressed_rag"):
        anchor = "memory_off" if label == "vs_memory_off" else "uncompressed_rag"
        keys = [f"delta_{label}_{f}" for f, _ in _DELTA_COLS]
        if not any(k in e for e in aggregate.values() for k in keys):
            continue
        lines += [f"## Δ {label} (anchor: `{anchor}`)", "",
                  "| Condition | " + " | ".join(c[1] for c in _DELTA_COLS) + " |",
                  "|" + "---|" * (len(_DELTA_COLS) + 1)]
        for cond, entry in aggregate.items():
            cells = [_fmt(entry.get(f"delta_{label}_{f}")) for f, _ in _DELTA_COLS]
            lines.append(f"| {cond} | " + " | ".join(cells) + " |")
        lines.append("")
    _write(path, "\n".join(lines))
    return path


def _write_stats_table(out_dir, stats_data, smoke) -> str:
    path = os.path.join(out_dir, "table_stats.md")
    lines = ["# Statistical comparisons", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not stats_data or not stats_data.get("comparisons"):
        lines += ["_EMPTY TEMPLATE — no comparisons available._", ""]
        _write(path, "\n".join(lines))
        return path
    lines += [f"Non-inferiority margin δ = {stats_data.get('noninferiority_delta')}", ""]
    lines += ["| Comparison | Δ task_score | 95% CI | perm p | non-inferior |",
              "|---|---|---|---|---|"]
    for name, comp in stats_data["comparisons"].items():
        diff = comp.get("paired_bootstrap_diff", {})
        perm = comp.get("paired_permutation", {})
        ni = comp.get("noninferiority", {})
        ci = f"[{_fmt(diff.get('lo'))}, {_fmt(diff.get('hi'))}]"
        lines.append(
            f"| {name} | {_fmt(diff.get('mean_diff'))} | {ci} | "
            f"{_fmt(perm.get('p_value'))} | {ni.get('non_inferior', 'n.a.')} |"
        )
    lines.append("")
    _write(path, "\n".join(lines))
    return path


# --- figures ----------------------------------------------------------------

def _figures(out_dir, aggregate, smoke) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        with open(os.path.join(out_dir, "FIGURES_SKIPPED.txt"), "w") as fh:
            fh.write("matplotlib not installed; install the [plot] extra to render figures.\n")
        return []
    if not aggregate:
        return []

    conds = list(aggregate.keys())
    made = []

    def _watermark(ax):
        if smoke:
            ax.text(0.5, 0.5, "SMOKE TEST", transform=ax.transAxes, fontsize=34,
                    color="red", alpha=0.20, ha="center", va="center", rotation=30,
                    fontweight="bold")

    def _bar(field, ylabel, fname, title):
        vals = [aggregate[c].get(field) for c in conds]
        if all(v is None for v in vals):
            return
        vals = [0 if v is None else v for v in vals]
        fig, ax = plt.subplots(figsize=(max(6, len(conds) * 1.1), 4))
        ax.bar(range(len(conds)), vals, color="steelblue")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title + ("  [SMOKE TEST]" if smoke else ""))
        _watermark(ax)
        fig.tight_layout()
        p = os.path.join(out_dir, fname)
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)

    # F1: quality-vs-bytes Pareto.
    xs = [aggregate[c].get("physical_serialized_bytes_mean") for c in conds]
    ys = [aggregate[c].get("task_score_mean") for c in conds]
    if any(x is not None for x in xs) and any(y is not None for y in ys):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c, x, y in zip(conds, xs, ys):
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=60)
            ax.annotate(c, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Physical serialized bytes (mean)")
        ax.set_ylabel("Task score (mean)")
        ax.set_title("F1 — Quality vs bytes (Pareto)" + ("  [SMOKE TEST]" if smoke else ""))
        _watermark(ax)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_pareto_quality_bytes.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)

    _bar("task_score_mean", "Task score", "fig_task_score.png", "Task score by condition")
    _bar("prompt_tokens_mean", "Prompt tokens", "fig_prompt_tokens.png", "Prompt tokens by condition")
    _bar("physical_serialized_bytes_mean", "Physical bytes", "fig_physical_bytes.png",
         "F3 — Physical serialized bytes by condition")

    # Tier occupancy stacked bar.
    tw = [aggregate[c].get("tier_working_mean") or 0 for c in conds]
    te = [aggregate[c].get("tier_episodic_mean") or 0 for c in conds]
    tsem = [aggregate[c].get("tier_semantic_mean") or 0 for c in conds]
    if any(tw) or any(te) or any(tsem):
        fig, ax = plt.subplots(figsize=(max(6, len(conds) * 1.1), 4))
        import numpy as np
        base = np.array(tw, dtype=float)
        ax.bar(range(len(conds)), tw, label="working")
        ax.bar(range(len(conds)), te, bottom=base, label="episodic")
        ax.bar(range(len(conds)), tsem, bottom=base + np.array(te), label="semantic")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Mean records")
        ax.set_title("Per-tier occupancy" + ("  [SMOKE TEST]" if smoke else ""))
        ax.legend(fontsize=8)
        _watermark(ax)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_tier_occupancy.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)
    return made


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def generate(run_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    aggregate = _load_json(os.path.join(run_dir, "aggregate.json"))
    stats_data = _load_json(os.path.join(run_dir, "stats.json"))
    manifest = _load_json(os.path.join(run_dir, "manifest.json"))
    smoke = _is_smoke(manifest, aggregate)

    made = []
    made.append(_write_main_table(out_dir, aggregate, smoke))
    made.append(_write_baseline_table(out_dir, aggregate, smoke))
    made.append(_write_deltas_table(out_dir, aggregate, smoke))
    made.append(_write_stats_table(out_dir, stats_data, smoke))
    figs = _figures(out_dir, aggregate, smoke)
    made.extend(figs)

    # Persist the run label (smoke watermark or PoC banner) as a standalone file.
    banner = _poc_banner(manifest)
    label = WATERMARK if smoke else (banner or "measured real-model run")
    _write(os.path.join(out_dir, "RUN_LABEL.txt"), label + "\n")

    # README-ish index of artifacts.
    index = ["# Paper artifacts", ""]
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
