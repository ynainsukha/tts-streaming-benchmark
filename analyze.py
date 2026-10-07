"""Turn results/<run_id>/*.jsonl into report.md + plots.

  python analyze.py results/<run_id> [results/<run_id2> ...]   # multiple runs are pooled
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RNG = np.random.default_rng(0)


def load(dirs, name):
    frames = []
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            df = pd.read_json(p, lines=True)
            df["run"] = Path(d).name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def boot_ci(x, fn, n=2000):
    x = np.asarray(x)
    if len(x) < 3:
        return (np.nan, np.nan)
    s = [fn(RNG.choice(x, len(x))) for _ in range(n)]
    return np.percentile(s, [2.5, 97.5])


def pct(q):
    return lambda x: np.percentile(x, q)


def latency_tables(lat):
    out = []
    for keys, g in lat.groupby(["model", "conn", "concurrency"]):
        t = g["ttfa_ms"]
        lo, hi = boot_ci(t, pct(50))
        out.append({
            "model": keys[0], "conn": keys[1], "conc": keys[2], "n": len(g),
            "TTFA p50": f"{t.median():.0f} [{lo:.0f}-{hi:.0f}]",
            "TTFA p95": f"{np.percentile(t, 95):.0f}",
            "TTFA p99": f"{np.percentile(t, 99):.0f}" if len(t) >= 100 else "n<100",
            "TTFA IQR": f"{np.percentile(t, 75) - np.percentile(t, 25):.0f}",
            "est. server TTFA p50": f"{(t - g['rtt_ms']).median():.0f}" if g["rtt_ms"].notna().any() else "-",
            "RTF p50": f"{g['rtf'].median():.2f}",
            "stall rate (0/100/250ms buf)": " / ".join(
                f"{(g[f'stalls_j{j}'] > 0).mean():.0%}" for j in (0, 100, 250)),
        })
    return pd.DataFrame(out)


def mwu(lat, a, b, col="ttfa_ms"):
    x, y = lat[lat.model == a][col], lat[lat.model == b][col]
    if len(x) < 3 or len(y) < 3:
        return None
    u, p = stats.mannwhitneyu(x, y)
    # Common-language effect size: P(random a request is faster than random b request)
    return {"p": p, "P(a<b)": (x.values[:, None] < y.values[None, :]).mean()}


def length_fit(lat):
    rows = []
    sweep = lat[lat.category == "length_sweep"]
    for m, g in sweep.groupby("model"):
        s_t = stats.linregress(g["chars"], g["ttfa_ms"])
        s_tot = stats.linregress(g["chars"], g["total_ms"])
        rows.append({"model": m,
                     "TTFA ms per 100 chars": f"{s_t.slope * 100:.1f}",
                     "TTFA intercept ms": f"{s_t.intercept:.0f}",
                     "total ms per 100 chars": f"{s_tot.slope * 100:.0f}",
                     "R2 total": f"{s_tot.rvalue ** 2:.2f}"})
    return pd.DataFrame(rows)


def cost_table(lat, rates):
    """rates: {model: credits per input character}, measured with before/after counter
    readings. Converted to cost per minute of generated audio using each model's own
    observed chars-per-second, since models can speak the same text at different speeds."""
    rows = []
    for m, rate in rates.items():
        g = lat[lat.model == m]
        if g.empty:
            continue
        cps = (g["chars"] / g["audio_s"]).median()
        rows.append({"model": m, "credits / char": f"{rate:.3f}",
                     "credits / 1K chars": f"{rate * 1000:.0f}",
                     "chars / s of audio": f"{cps:.1f}",
                     "credits / min of audio": f"{rate * cps * 60:.1f}",
                     "TTFA p50 (ms)": f"{g['ttfa_ms'].median():.0f}"})
    return pd.DataFrame(rows)


def plots(lat, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for (m, c), g in lat.groupby(["model", "conn"]):
        x = np.sort(g["ttfa_ms"])
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=f"{m} ({c})")
    ax.set(xlabel="time to first audio (ms)", ylabel="fraction of requests", title="TTFA ECDF")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "ttfa_ecdf.png", dpi=150)

    sweep = lat[lat.category == "length_sweep"]
    if len(sweep):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, g in sweep.groupby("model"):
            med = g.groupby("chars")[["ttfa_ms", "total_ms"]].median()
            ax.plot(med.index, med.ttfa_ms, "o-", label=f"{m} TTFA")
            ax.plot(med.index, med.total_ms, "s--", label=f"{m} total")
        ax.set(xlabel="input characters", ylabel="ms (median)", title="Latency vs input length")
        ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "length_sweep.png", dpi=150)

    # One representative streaming timeline per model: audio received vs audio played.
    fig, ax = plt.subplots(figsize=(7, 4))
    src = sweep if len(sweep) else lat
    for m, g in src.groupby("model"):
        r = g.loc[g["chars"].idxmax()]
        arr, dur = np.array(r.chunk_arrivals_ms), np.cumsum(r.chunk_audio_ms)
        ax.step(arr, dur, where="post", label=f"{m}: audio received")
        ax.plot([arr[0], arr[0] + dur[-1]], [0, dur[-1]], ":", label=f"{m}: playback (1x)")
    ax.set(xlabel="wall clock (ms)", ylabel="audio (ms)", title="Generation vs playback (longest input)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "stream_timeline.png", dpi=150)


def quality_tables(q):
    cols = [c for c in ["wer", "cer", "utmos", "f0_range_st", "f0_std_st", "energy_std_db",
                        "speech_rate_wps", "pause_count", "lead_silence_ms", "spk_sim"] if c in q]
    overall = q.groupby("model")[cols].agg(["mean", "std"]).round(3)
    overall.columns = [f"{a} {b}" for a, b in overall.columns]

    qs = q[q.category.str.startswith("question")]
    question = qs.groupby("model")[[c for c in ["final_rise_st", "final_slope_st_per_s"] if c in qs]].agg(["mean", "std"]).round(2)

    by_cat = q.pivot_table(index="category", columns="model",
                           values=[c for c in ["wer", "f0_range_st", "utmos"] if c in q], aggfunc="mean").round(3)
    return overall, question, by_cat


def paired(q, a, b, metrics):
    """Per-item mean for each model, then mean paired difference (a - b) with bootstrap CI."""
    m = q.groupby(["item_id", "model"])[metrics].mean().unstack("model")
    rows = []
    for met in metrics:
        if (met, a) not in m or (met, b) not in m:
            continue
        d = (m[(met, a)] - m[(met, b)]).dropna()
        if len(d) < 3:
            continue
        lo, hi = boot_ci(d, np.mean)
        w = stats.wilcoxon(d) if (d != 0).any() else None
        rows.append({"metric": met, f"mean({a} - {b})": f"{d.mean():.3f}", "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                     "items": len(d), "wilcoxon p": f"{w.pvalue:.3g}" if w else "-"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cost", nargs="*", default=[], metavar="MODEL=RATE",
                    help="measured credits per character, e.g. eleven_v4=0.109")
    a = ap.parse_args()
    out = Path(a.out or a.runs[-1])
    out.mkdir(parents=True, exist_ok=True)

    lat = load(a.runs, "latency.jsonl")
    lat = lat[lat.ok == True].copy()
    q = load(a.runs, "quality.jsonl")
    models = sorted(lat.model.unique())

    md = ["# Results\n", f"Runs: {', '.join(Path(r).name for r in a.runs)}  \n"
          f"Requests: {len(lat)}  \nModels: {', '.join(models)}\n",
          "## Latency\n", "TTFA in ms; p50 shown with bootstrap 95% CI.\n",
          latency_tables(lat).to_markdown(index=False), "\n"]
    if len(models) >= 2:
        r = mwu(lat, models[0], models[1])
        if r:
            md.append(f"\nMann-Whitney U ({models[0]} vs {models[1]}, TTFA): p = {r['p']:.2g}; "
                      f"P({models[0]} request faster) = {r['P(a<b)']:.2f}\n")
    lf = length_fit(lat)
    if len(lf):
        md += ["\n## Scaling with input length\n", lf.to_markdown(index=False),
               "\n\n![length](length_sweep.png)\n"]
    md += ["\n![ecdf](ttfa_ecdf.png)\n\n![timeline](stream_timeline.png)\n"]
    if a.cost:
        rates = {k: float(v) for k, v in (c.split("=", 1) for c in a.cost)}
        md += ["\n## Cost\n", "Measured from account credit counter before/after single-model runs; "
               "plan- and date-specific, not list pricing.\n\n", cost_table(lat, rates).to_markdown(index=False), "\n"]

    if q is not None:
        overall, question, by_cat = quality_tables(q)
        md += ["\n## Quality (objective)\n", overall.to_markdown(), "\n\n### Question intonation\n",
               "final_rise_st: pitch of the last 150 ms relative to the utterance median (semitones). "
               "Positive = rising ending.\n\n", question.to_markdown(), "\n\n### By category\n", by_cat.to_markdown()]
        if len(models) >= 2:
            metrics = [c for c in ["wer", "cer", "utmos", "f0_range_st", "final_rise_st",
                                   "speech_rate_wps", "spk_sim"] if c in q]
            md += ["\n\n### Paired comparison (same input text)\n", paired(q, models[0], models[1], metrics).to_markdown(index=False)]

    plots(lat, out)
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
