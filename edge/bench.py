# plot_simple_mimic.py
# Two simple bars + concise footer, with solid layout and readable styling.

import os
import math
import textwrap
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
MIMIC_SUMMARY = "results/mimic_summary.csv"   
BENCH_SUMMARY = "results/bench_summary.csv"   
OUT_DIR       = "results/plots"
STREAM_PREF   = "med"                         
FIGSIZE       = (11, 6.5)                     
STYLE         = "seaborn-v0_8-whitegrid"
ANNOT_FONTSZ  = 11
TITLE_FONTSZ  = 16
LABEL_FONTSZ  = 12
TICK_FONTSZ   = 11

# Helpers
def pick_stream(df: pd.DataFrame, prefer: str = "med") -> pd.DataFrame:
    if "stream" not in df.columns:
        return df.copy()
    if (df["stream"] == prefer).any():
        return df[df["stream"] == prefer].copy()
    # fallback: whichever stream has most rows
    top = df["stream"].value_counts().idxmax()
    return df[df["stream"] == top].copy()

def best_variant_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model family, keep the row with the smallest avg latency.
    If tie, take the one with smallest p95.
    """
    # guard for missing columns
    if "lat_ms_mean" not in df.columns:
        raise ValueError("mimic_summary.csv must contain 'lat_ms_mean'.")
    if "model" not in df.columns:
        raise ValueError("mimic_summary.csv must contain 'model'.")
    if "variant" not in df.columns:
        df["variant"] = ""
    if "lat_ms_p95" not in df.columns:
        df["lat_ms_p95"] = np.nan

    # sort by (mean, then p95) so first row per model is the best
    df_sorted = df.sort_values(["model", "lat_ms_mean", "lat_ms_p95"], ascending=[True, True, True])
    keep_idx = df_sorted.groupby("model", as_index=False).head(1).index
    best = df_sorted.loc[keep_idx].copy()

    # stable model order: by ascending avg latency
    best = best.sort_values("lat_ms_mean", ascending=True)
    return best

def safe_col(d: pd.Series, col: str, default=np.nan):
    return d[col] if col in d and pd.notna(d[col]) else default

def compute_footer(best: pd.DataFrame, bench_summary_path: str) -> str:
    # RAM peak range
    ram_min = best["rss_mb_max"].min() if "rss_mb_max" in best.columns else np.nan
    ram_max = best["rss_mb_max"].max() if "rss_mb_max" in best.columns else np.nan
    ram_note = ""
    if pd.notna(ram_min) and pd.notna(ram_max):
        ram_note = f"RAM (peak): {ram_min:.1f}–{ram_max:.1f} MB"

    # container image size
    img_note = ""
    if os.path.exists(bench_summary_path):
        try:
            bs = pd.read_csv(bench_summary_path)
            if "container_image_mb" in bs.columns and bs["container_image_mb"].notna().any():
                img_mb = float(bs["container_image_mb"].dropna().iloc[0])
                img_note = f"Image: ~{img_mb:.0f} MB"
        except Exception:
            pass

    # real-time safety check (no overruns) using p95 vs tick
    # If every row satisfies p95 < tick*1000 -> "No crashes/overruns."
    rt_note = ""
    if {"lat_ms_p95","tick_seconds"}.issubset(best.columns):
        safe_all = (best["lat_ms_p95"] < (best["tick_seconds"] * 1000.0)).all()
        rt_note = "No crashes/overruns." if safe_all else "Check overruns in logs."

    # combine neatly
    bits = [x for x in [ram_note, img_note, rt_note] if x]
    if not bits:
        return ""
    return " • ".join(bits)

def annotate_bars(ax, fmt="{:.2f}", fontsize=ANNOT_FONTSZ, pad=3):
    for p in ax.patches:
        h = p.get_height()
        if math.isnan(h):
            continue
        ax.annotate(fmt.format(h),
                    (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom",
                    fontsize=fontsize, xytext=(0, pad),
                    textcoords="offset points")

def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(MIMIC_SUMMARY):
        raise FileNotFoundError(f"Cannot find {MIMIC_SUMMARY}")

    df = pd.read_csv(MIMIC_SUMMARY)

    # pick stream & best variant per family
    sel = pick_stream(df, STREAM_PREF)
    best = best_variant_per_model(sel)

    # Construct display name: Model (variant)
    best = best.copy()
    best["display"] = best.apply(
        lambda r: f"{r['model']} {r['variant']}".strip(), axis=1
    )

    # Build the 2 metrics we need
    # Avg latency (ms)
    if "lat_ms_mean" not in best.columns:
        raise ValueError("lat_ms_mean missing in mimic_summary.csv")
    # Avg CPU (%) (approx)
    if "cpu_util_pct_appx" not in best.columns:
        best["cpu_util_pct_appx"] = np.nan  # still plot; will just be empty

    # Order by latency ascending for both charts
    best = best.sort_values("lat_ms_mean", ascending=True).reset_index(drop=True)

    # --- Plot ---
    plt.style.use(STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

    # Bar 1: Avg latency
    ax1.bar(best["display"], best["lat_ms_mean"])
    ax1.set_title("Avg latency per prediction (ms)", fontsize=TITLE_FONTSZ, pad=12)
    ax1.set_ylabel("Milliseconds (ms)", fontsize=LABEL_FONTSZ)
    ax1.tick_params(axis="x", labelrotation=20, labelsize=TICK_FONTSZ)
    ax1.tick_params(axis="y", labelsize=TICK_FONTSZ)
    annotate_bars(ax1, fmt="{:.2f}")

    # Bar 2: Avg CPU
    ax2.bar(best["display"], best["cpu_util_pct_appx"])
    ax2.set_title("Avg CPU usage (%)", fontsize=TITLE_FONTSZ, pad=12)
    ax2.set_ylabel("CPU (%)", fontsize=LABEL_FONTSZ)
    ax2.tick_params(axis="x", labelrotation=20, labelsize=TICK_FONTSZ)
    ax2.tick_params(axis="y", labelsize=TICK_FONTSZ)
    annotate_bars(ax2, fmt="{:.1f}")

    # Footer notes under the figure
    footer = compute_footer(best, BENCH_SUMMARY)
    if footer:
        fig.suptitle("", y=0.98)  # keep headroom clean
        # Add a centered footer line just below the axes
        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=12, alpha=0.9)

    # Save
    out_path = os.path.join(OUT_DIR, f"simple_comparison_{STREAM_PREF}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also print a tiny textual summary in console (nice for logs)
    cols = ["model","variant","lat_ms_mean","cpu_util_pct_appx","rss_mb_max","lat_ms_p95","tick_seconds"]
    cols = [c for c in cols if c in best.columns]
    print("\n=== SIMPLE COMPARISON ROWS (best variant per model) ===")
    print(best[cols].to_string(index=False))

if __name__ == "__main__":
    main()
