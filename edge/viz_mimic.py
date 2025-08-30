# plot_simple_mimic.py
# Two simple bars + (1) all variants per model + (2) RAM (peak) best variants
# + (3) p95 latency (best variants) + (4) Money table
# Fancy visuals + consistent, color-blind friendly palette.

import os
import math
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Config
MIMIC_SUMMARY = "results/mimic_summary.csv"
BENCH_SUMMARY = "results/bench_summary.csv"   
OUT_DIR       = "results/plots"
STREAM_PREF   = "med"                         

FIGSIZE_MAIN  = (13.8, 7.6)                  
FIGSIZE_GRID  = (14.2, 8.6)                   
FIGSIZE_RAM   = (12.6, 6.4)                   
FIGSIZE_P95   = (12.6, 6.4)                   
FIGSIZE_TABLE = (12.0, 6.0)                   

BASE_STYLE    = "seaborn-v0_8-whitegrid"
TITLE_FONTSZ  = 19
LABEL_FONTSZ  = 13
TICK_FONTSZ   = 11
ANNOT_FONTSZ  = 11

# Palette

MODEL_COLORS = {
    "LiteFormer": "#0072B2",   
    "LiPFormer":  "#E69F00",  
    "TinyFormer": "#009E73",   
    "LSTM":       "#D55E00",   
}
FALLBACK_COLOR = "#6c757d"     


# Styling helpers
def apply_theme():
    plt.style.use(BASE_STYLE)
    plt.rcParams.update({
        "figure.facecolor": "#f7f7f7",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#D0D0D0",
        "axes.linewidth": 1.0,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": TITLE_FONTSZ,
        "axes.labelsize": LABEL_FONTSZ,
        "xtick.labelsize": TICK_FONTSZ,
        "ytick.labelsize": TICK_FONTSZ,
        "legend.frameon": False,
    })


def annotate_bars(ax, fmt="{:.2f}", fontsize=ANNOT_FONTSZ, pad_frac=0.03):
    ymax = ax.get_ylim()[1]
    dy = ymax * pad_frac
    for p in ax.patches:
        h = p.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width()/2, h + dy),
            ha="center", va="bottom",
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cfcfcf", alpha=0.95),
            path_effects=[pe.withStroke(linewidth=1.0, foreground="white")]
        )


def tidy_axes(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CFCFCF")
        ax.spines[side].set_linewidth(1.0)


# Data helpers
def pick_stream(df: pd.DataFrame, prefer: str = "med") -> pd.DataFrame:
    if "stream" not in df.columns:
        return df.copy()
    s = df["stream"].astype(str).str.lower()
    prefer = (prefer or "").lower()
    if (s == prefer).any():
        return df[s == prefer].copy()
    top = s.value_counts().idxmax()
    return df[s == top].copy()


def _friendly_variant(v: str) -> str:
    v = str(v or "").strip().replace("_", "-")
    return v.replace("int8-dynamic", "int8-dyn")


def best_variant_per_model(df: pd.DataFrame) -> pd.DataFrame:
    if "lat_ms_mean" not in df.columns:
        raise ValueError("mimic_summary.csv must contain 'lat_ms_mean'.")
    if "model" not in df.columns:
        raise ValueError("mimic_summary.csv must contain 'model'.")
    if "variant" not in df.columns:
        df["variant"] = ""
    if "lat_ms_p95" not in df.columns:
        df["lat_ms_p95"] = np.nan

    df_sorted = df.sort_values(
        ["model", "lat_ms_mean", "lat_ms_p95"],
        ascending=[True, True, True]
    )
    best = df_sorted.drop_duplicates(subset=["model"], keep="first").copy()
    best = best.sort_values("lat_ms_mean", ascending=True)
    return best


def compute_footer(best: pd.DataFrame, bench_summary_path: str) -> str:
    ram_note = ""
    if "rss_mb_max" in best.columns and best["rss_mb_max"].notna().any():
        ram_min = best["rss_mb_max"].min()
        ram_max = best["rss_mb_max"].max()
        ram_note = f"RAM (peak): {ram_min:.1f}–{ram_max:.1f} MB"

    img_note = ""
    if os.path.exists(bench_summary_path):
        try:
            bs = pd.read_csv(bench_summary_path)
            if "container_image_mb" in bs.columns and bs["container_image_mb"].notna().any():
                img_mb = float(bs["container_image_mb"].dropna().iloc[0])
                img_note = f"Image: ~{img_mb:.0f} MB"
        except Exception:
            pass

    rt_note = ""
    if {"lat_ms_p95","tick_seconds"}.issubset(best.columns):
        safe_all = (best["lat_ms_p95"] < (best["tick_seconds"] * 1000.0)).all()
        rt_note = "No crashes/overruns." if safe_all else "Check overruns in logs."

    bits = [x for x in [ram_note, img_note, rt_note] if x]
    return " • ".join(bits)


# Plots
def plot_variants_per_model(sel: pd.DataFrame, out_path: str):
    if not {"model","variant","lat_ms_mean"}.issubset(sel.columns):
        print("Skipping variants-per-model plot: required columns missing.")
        return

    models = sorted(sel["model"].unique().tolist())
    if not models:
        print("No models to plot for variants-per-model.")
        return

    ymax = sel["lat_ms_mean"].max() * 1.25
    n = len(models)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    apply_theme()
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 8.6), squeeze=False)
    ax_flat = axes.ravel()

    for i, m in enumerate(models):
        ax = ax_flat[i]
        dfm = sel[sel["model"] == m].copy().sort_values("lat_ms_mean")
        dfm["disp"] = dfm["variant"].apply(_friendly_variant)
        # single hue per model family for cohesiveness
        base = MODEL_COLORS.get(m, FALLBACK_COLOR)
        ax.bar(dfm["disp"], dfm["lat_ms_mean"], color=base, edgecolor="#555", linewidth=0.7)

        ax.set_title(f"{m} — variants (avg latency, ms)", pad=10)
        ax.set_ylabel("Avg latency (ms)")
        ax.tick_params(axis="x", labelrotation=18)
        tidy_axes(ax)
        ax.set_ylim(0, ymax if np.isfinite(ymax) and ymax > 0 else 1)
        annotate_bars(ax, fmt="{:.2f}", fontsize=ANNOT_FONTSZ)

    for j in range(i + 1, len(ax_flat)):
        fig.delaxes(ax_flat[j])

    fig.suptitle(f"All variants performance per model — stream={STREAM_PREF}", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.92, hspace=0.35, wspace=0.28)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot_ram_best_variants(best: pd.DataFrame, out_path: str):
    if "rss_mb_max" not in best.columns:
        print("Skipping RAM plot: 'rss_mb_max' missing.")
        return

    apply_theme()
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_RAM)

    labels = best.apply(lambda r: f"{r['model']} ({_friendly_variant(r['variant'])})", axis=1)
    colors = [MODEL_COLORS.get(m, FALLBACK_COLOR) for m in best["model"]]
    ax.bar(labels, best["rss_mb_max"], color=colors, edgecolor="#555", linewidth=0.7)

    ax.set_title("RAM peak (MB) — best variant per model", pad=12)
    ax.set_ylabel("RAM (MB)")
    ax.tick_params(axis="x", labelrotation=16)
    tidy_axes(ax)
    ax.set_ylim(0, max(best["rss_mb_max"]) * 1.18 if len(best) else 1)
    annotate_bars(ax, fmt="{:.1f}")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.90)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


# plots
def plot_p95_best_variants(best: pd.DataFrame, out_path: str):
    """Bar: p95 latency (ms) — best variant per model, with 1s tick guide."""
    if "lat_ms_p95" not in best.columns:
        print("Skipping p95 plot: 'lat_ms_p95' missing.")
        return

    apply_theme()
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_P95)

    labels = best.apply(lambda r: f"{r['model']} ({_friendly_variant(r['variant'])})", axis=1)
    colors = [MODEL_COLORS.get(m, FALLBACK_COLOR) for m in best["model"]]
    p95_vals = best["lat_ms_p95"].astype(float)

    ax.bar(labels, p95_vals, color=colors, edgecolor="#555", linewidth=0.7)
    ax.set_title("p95 latency (ms) — best variant per model", pad=12)
    ax.set_ylabel("Milliseconds (ms)")
    ax.tick_params(axis="x", labelrotation=16)
    tidy_axes(ax)

    ymax = (np.nanmax(p95_vals) if len(p95_vals) else 1.0) * 1.25
    ax.set_ylim(0, ymax if np.isfinite(ymax) and ymax > 0 else 1)

    # Tick=1s guide (if available)
    if "tick_seconds" in best.columns and best["tick_seconds"].notna().any():
        thr = float(best["tick_seconds"].dropna().iloc[0]) * 1000.0
        ax.axhline(thr, linestyle="--", color="#666", lw=1.2, alpha=0.75)
        ax.text(0.98, min(thr / ax.get_ylim()[1] + 0.02, 0.95),
                f"tick = {int(thr)} ms",
                ha="right", va="bottom", color="#555", fontsize=10, transform=ax.transAxes)

    annotate_bars(ax, fmt="{:.3f}")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.90)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


def save_money_table(best: pd.DataFrame, out_path: str):
    """Render a compact 'money table' image for best variant per model."""
    df = best.copy()

    # Columns
    df["Model"] = df["model"]
    df["Variant"] = df["variant"].apply(_friendly_variant)
    df["Avg ms"] = df.get("lat_ms_mean", np.nan)
    df["p50 ms"] = df.get("lat_ms_p50", np.nan)
    df["p95 ms"] = df.get("lat_ms_p95", np.nan)
    df["Jitter"] = df["p95 ms"] / df["p50 ms"]
    if "tick_seconds" in df.columns and df["tick_seconds"].notna().any():
        df["Headroom %"] = 100.0 * (1.0 - (df["p95 ms"] / (df["tick_seconds"] * 1000.0)))
    else:
        df["Headroom %"] = np.nan
    df["CPU % (avg)"] = df.get("cpu_util_pct_appx", np.nan)
    df["Peak RAM MB"] = df.get("rss_mb_max", np.nan)
    df["Status"] = np.where(df["Headroom %"] > 0, "OK", "Check")

    show_cols = ["Model","Variant","Avg ms","p50 ms","p95 ms","Jitter","Headroom %","CPU % (avg)","Peak RAM MB","Status"]
    df_show = df[show_cols].copy()
    df_show = df_show.round({"Avg ms":3, "p50 ms":3, "p95 ms":3, "Jitter":2, "Headroom %":1, "CPU % (avg)":1, "Peak RAM MB":1})

    apply_theme()
    # Figure height scales with rows
    fig_h = 0.8 + 0.5 * max(1, len(df_show))
    fig, ax = plt.subplots(figsize=(FIGSIZE_TABLE[0], fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=df_show.values,
                   colLabels=df_show.columns,
                   loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.3)

    # header styling
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F5F5F5")

    fig.suptitle(f"Edge mimic — best variants (stream={STREAM_PREF})", fontsize=TITLE_FONTSZ, y=0.98)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


#  Main summary (two bars) 
def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(MIMIC_SUMMARY):
        raise FileNotFoundError(f"Cannot find {MIMIC_SUMMARY}")

    df = pd.read_csv(MIMIC_SUMMARY)
    sel = pick_stream(df, STREAM_PREF)
    best = best_variant_per_model(sel).reset_index(drop=True)

    best["display"] = best.apply(
        lambda r: f"{r['model']} ({_friendly_variant(r['variant'])})", axis=1
    )
    if "cpu_util_pct_appx" not in best.columns:
        best["cpu_util_pct_appx"] = np.nan

    labels = best["display"].tolist()
    colors = [MODEL_COLORS.get(m, FALLBACK_COLOR) for m in best["model"]]

    # --- Figure 1: Avg latency + Avg CPU ---
    apply_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_MAIN, constrained_layout=False)

    # Avg latency
    ax1.bar(labels, best["lat_ms_mean"], color=colors, edgecolor="#555", linewidth=0.7)
    ax1.set_title("Avg latency per prediction (ms)", pad=12)
    ax1.set_ylabel("Milliseconds (ms)")
    ax1.tick_params(axis="x", labelrotation=16)
    tidy_axes(ax1)
    ax1.set_ylim(0, max(best["lat_ms_mean"]) * 1.25 if len(best) else 1)
    annotate_bars(ax1, fmt="{:.2f}")

    # Avg CPU
    ax2.bar(labels, best["cpu_util_pct_appx"], color=colors, edgecolor="#555", linewidth=0.7)
    ax2.set_title("Avg CPU usage (%)", pad=12)
    ax2.set_ylabel("CPU (%)")
    ax2.tick_params(axis="x", labelrotation=16)
    tidy_axes(ax2)
    ymax_cpu = np.nanmax(best["cpu_util_pct_appx"]) if best["cpu_util_pct_appx"].notna().any() else 1.0
    ax2.set_ylim(0, ymax_cpu * 1.25)
    annotate_bars(ax2, fmt="{:.1f}")

    # Footer
    footer = compute_footer(best, BENCH_SUMMARY)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.22, top=0.90, wspace=0.26)
    if footer:
        fig.lines.extend([plt.Line2D([0.06, 0.98], [0.12, 0.12], transform=fig.transFigure, color="#E0E0E0", lw=1)])
        fig.text(0.5, 0.08, footer, ha="center", va="bottom", fontsize=12.5, color="#333333")

    out_main = os.path.join(OUT_DIR, f"simple_comparison_{STREAM_PREF}.png")
    fig.savefig(out_main, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_main}")

    # --- Figure 2: all variants performance per model ---
    out_variants = os.path.join(OUT_DIR, f"variants_per_model_{STREAM_PREF}.png")
    plot_variants_per_model(sel, out_variants)

    # --- Figure 3: RAM (peak) best variants per model ---
    out_ram = os.path.join(OUT_DIR, f"ram_best_variants_{STREAM_PREF}.png")
    plot_ram_best_variants(best, out_ram)

    # --- Figure 4: p95 latency (best variants) ---
    out_p95 = os.path.join(OUT_DIR, f"p95_best_variants_{STREAM_PREF}.png")
    plot_p95_best_variants(best, out_p95)

    # --- Figure 5: Money table (best variants) ---
    out_money = os.path.join(OUT_DIR, f"money_table_{STREAM_PREF}.png")
    save_money_table(best, out_money)

    # Console summary
    cols = ["model","variant","lat_ms_mean","cpu_util_pct_appx","rss_mb_max","lat_ms_p95","tick_seconds"]
    cols = [c for c in cols if c in best.columns]
    print("\n=== SIMPLE COMPARISON ROWS (best variant per model) ===")
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
