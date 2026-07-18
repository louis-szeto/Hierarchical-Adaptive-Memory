"""Paper-ready artifacts for the stage-D KV experiment, from *actual* run outputs
only. Headline = HAM/full bytes-ratio and quality vs redundancy (the slope proves
'frequency'). Mock output watermarked ``SMOKE TEST``; empty dirs -> EMPTY TEMPLATE."""

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


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def _fmt(v, nd=3):
    if v is None:
        return "n.a."
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _is_smoke(manifest) -> bool:
    return bool(manifest.get("is_smoke")) if manifest else False


def _poc_banner(manifest):
    if not manifest or manifest.get("is_smoke"):
        return None
    notes = (manifest.get("config", {}) or {}).get("notes", "") or ""
    if "PROOF OF CONCEPT" in notes.upper():
        m = (manifest.get("config", {}) or {}).get("backend", {}).get("model_id", "?")
        return f"{POC_BANNER} (model: {m})"
    return None


def _write_redundancy_table(out_dir, aggregate, smoke) -> str:
    path = os.path.join(out_dir, "table_redundancy.md")
    lines = ["# Table KV1 — KV-cache cost vs full, by redundancy & strength", "",
             "Ratios are condition / full_kv at the same redundancy. bytes < 1.0 = "
             "smaller; quality_delta ~ 0 = iso-quality. HAM's advantage should GROW with "
             "redundancy (the proof frequency is the mechanism).", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found. Run a kvbench experiment first._", ""]
        _write(path, "\n".join(lines))
        return path
    lines += ["| redundancy | condition | keep_ratio | bytes_ratio | quality_delta |",
              "|---|---|---|---|---|"]
    for e in sorted(aggregate.values(), key=lambda x: (x["redundancy"], x["condition"], -(x["keep_ratio"]))):
        lines.append(f"| {e['redundancy']} | {e['condition']} | {e['keep_ratio']} | "
                     f"{_fmt(e['bytes_ratio_vs_full'])} | "
                     f"{_fmt(e['quality_delta_vs_full'])} |")
    lines.append("")
    _write(path, "\n".join(lines))
    return path


def _write_quality_bytes_table(out_dir, aggregate, smoke) -> str:
    path = os.path.join(out_dir, "table_quality_bytes.md")
    lines = ["# Table KV2 — Quality vs byte-honest KV size (Pareto sweep)", ""]
    if smoke:
        lines += [f"> **{WATERMARK}**", ""]
    if not aggregate:
        lines += ["_EMPTY TEMPLATE — no run data found._", ""]
        _write(path, "\n".join(lines))
        return path
    lines += ["| redundancy | condition | keep_ratio | agreement | kv_bytes |",
              "|---|---|---|---|---|"]
    for e in sorted(aggregate.values(), key=lambda x: (x["redundancy"], x["condition"], -(x["keep_ratio"]))):
        lines.append(f"| {e['redundancy']} | {e['condition']} | {e['keep_ratio']} | "
                     f"{_fmt(e['quality_agreement_mean'])} | {int(e['kv_bytes_mean'])} |")
    lines.append("")
    _write(path, "\n".join(lines))
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

    # Pareto: quality (agreement) vs bytes, per condition, across the keep_ratio sweep,
    # one subplot per redundancy level. HAM should dominate (high quality, low bytes)
    # increasingly as redundancy rises.
    reds = sorted({e["redundancy"] for e in aggregate.values()})
    by: dict[tuple, list[tuple[float, float]]] = defaultdict(list)  # (red, cond) -> (bytes, agreement)
    for e in aggregate.values():
        if e["bytes_ratio_vs_full"] is not None:
            by[(e["redundancy"], e["condition"])].append((e["kv_bytes_mean"], e["quality_agreement_mean"]))
    if reds and by:
        n = len(reds)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
        for i, red in enumerate(reds):
            ax = axes[0][i]
            for cond in sorted({c for (r, c) in by if r == red}):
                pts = sorted(by[(red, cond)])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=cond)
            ax.set_xlabel("KV bytes (byte-honest)")
            ax.set_ylabel("next-token agreement vs full_kv")
            ax.set_title(f"redundancy={red}" + ("  [SMOKE]" if smoke else ""))
            ax.legend(fontsize=6)
            _wm(ax)
        fig.suptitle("Quality-vs-bytes Pareto (HAM should dominate as redundancy rises)")
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_pareto_quality_bytes.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)
    return made


def generate(run_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    aggregate = _load_json(os.path.join(run_dir, "aggregate.json"))
    manifest = _load_json(os.path.join(run_dir, "manifest.json"))
    smoke = _is_smoke(manifest)
    made = [_write_redundancy_table(out_dir, aggregate, smoke),
            _write_quality_bytes_table(out_dir, aggregate, smoke)]
    made.extend(_figures(out_dir, aggregate, smoke))
    banner = _poc_banner(manifest)
    label = WATERMARK if smoke else (banner or "measured kvbench run")
    _write(os.path.join(out_dir, "RUN_LABEL.txt"), label + "\n")
    index = ["# KV-cache compression (stage D) artifacts", ""]
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
