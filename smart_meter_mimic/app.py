# app.py
import os, math
from datetime import datetime
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


try:
    import psutil
    HAVE_PSUTIL = True
except Exception:
    HAVE_PSUTIL = False


# Environment & constants

LOG_CSV     = os.getenv("LOG_CSV", "./logs/live_predictions.csv")
REFRESH_MS  = int(os.getenv("REFRESH_MS", "120000"))   # 2 min auto refresh
H           = int(os.getenv("HORIZON", "12"))
TZ_LABEL    = os.getenv("TZ_LABEL", "Europe/Dublin")

# Gauge scale & threshold 

GAUGE_MAX_6H   = float(os.getenv("GAUGE_MAX_6H", "8.0"))
GAUGE_THRESH_6H= float(os.getenv("GAUGE_THRESH_6H", "4.0"))

# Confidence band & tariff settings
CONF_HISTORY_N = int(os.getenv("CONF_HISTORY_N", "500"))  # past records for band calc
CONF_Z         = float(os.getenv("CONF_Z", "1.64"))       # 90% ≈ 1.64
TARIFF_PER_KWH = float(os.getenv("TARIFF_PER_KWH", "0.30"))

# Status thresholds (green/amber/red)
ERR_OK, ERR_WARN           = float(os.getenv("ERR_OK", "5")),   float(os.getenv("ERR_WARN", "10"))   # %
LAT_OK, LAT_WARN           = float(os.getenv("LAT_OK", "50")),  float(os.getenv("LAT_WARN", "200"))  # ms
CPU_OK, CPU_WARN           = float(os.getenv("CPU_OK", "40")),  float(os.getenv("CPU_WARN", "70"))   # %
RAM_OK, RAM_WARN           = float(os.getenv("RAM_OK", "60")),  float(os.getenv("RAM_WARN", "80"))   # %

st.set_page_config(page_title="Smart Meter — Live Forecast", layout="wide")

# Permanent dark styling

st.markdown(
    """
    <style>
      html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f16 !important;
        color: #e5e7eb !important;
      }
      .stMetric { background:rgba(255,255,255,0.03); border-radius:10px; padding:8px; }
      .status-box {
        border-radius: 10px; padding: 12px 14px; margin: 4px 0;
        display: flex; align-items: center; gap: 10px;
        background: rgba(255,255,255,0.03);
      }
      .status-dot { width: 10px; height: 10px; border-radius: 50%; }
      .status-label { font-weight: 600; }
      .status-value { font-variant-numeric: tabular-nums; }
      hr { border-top: 1px solid #1f2a37; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Auto refresh via JS timer

st.markdown(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {REFRESH_MS});
    </script>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Meter — Live Forecast (30-minute cadence)")

# Utilities

def fmt_ts(ts):
    try:
        dt = pd.to_datetime(ts)
        return dt.strftime("%H:%M")
    except Exception:
        return str(ts)

def altair_gauge(value: float,
                 max_value: float,
                 threshold: float,
                 title: str = "kWh (next 6h)",
                 width: int = 320,
                 height: int = 260) -> alt.Chart:
    v = max(0.0, min(float(value), float(max_value)))
    pct = v / max_value if max_value > 0 else 0.0

    inner_r = 56
    outer_r = 92

    data_bg  = pd.DataFrame({"start": [0], "end": [2 * math.pi]})
    data_val = pd.DataFrame({"start": [0], "end": [2 * math.pi * pct]})

    color_val = "#10b981" if v <= threshold else "#ef4444"

    base = (
        alt.Chart(data_bg)
        .mark_arc(innerRadius=inner_r, outerRadius=outer_r)
        .encode(theta=alt.Theta("end:Q", stack=None), color=alt.value("#2b3447"))
    )

    valc = (
        alt.Chart(data_val)
        .mark_arc(innerRadius=inner_r, outerRadius=outer_r)
        .encode(theta=alt.Theta("end:Q", stack=None), color=alt.value(color_val))
    )

    value_text = (
        alt.Chart(pd.DataFrame({"v": [v]}))
        .mark_text(fontSize=30, fontWeight="bold", color="#f8fafc")
        .encode(text=alt.Text("v:Q", format=".2f"))
    )

    label_text = (
        alt.Chart(pd.DataFrame({"t": [title]}))
        .mark_text(dy=26, color="#9aa3af")
        .encode(text="t:N")
    )

    return (
        (base + valc + value_text + label_text)
        .configure_view(strokeWidth=0)
        .properties(width=width, height=height,
                    padding={"top": 34, "bottom": 10, "left": 10, "right": 10})
    )

def classify_gar(value, ok, warn, invert=False):
    v = float(value)
    if np.isnan(v):
        return "#9aa3af", "n/a"
    if not invert:
        if v <= ok:      return "#10b981", "good"
        if v <= warn:    return "#f59e0b", "watch"
        return "#ef4444", "high"
    else:
        if v >= ok:      return "#10b981", "good"
        if v >= warn:    return "#f59e0b", "watch"
        return "#ef4444", "low"

def status_box(label, value_str, color_hex):
    return f"""
    <div class="status-box">
        <div class="status-dot" style="background:{color_hex};"></div>
        <div class="status-label">{label}</div>
        <div class="status-value" style="margin-left:auto;">{value_str}</div>
    </div>
    """

def safe_pct_err(actual_sum, pred_sum):
    if actual_sum is None or np.isnan(actual_sum) or actual_sum <= 1e-9:
        return np.nan
    return 100.0 * (pred_sum - actual_sum) / actual_sum

# Track averages across the session
if "latency_hist" not in st.session_state:
    st.session_state.latency_hist = []
if "cpu_hist" not in st.session_state:
    st.session_state.cpu_hist = []
if "ram_hist" not in st.session_state:
    st.session_state.ram_hist = []

# Loading live log

if not os.path.exists(LOG_CSV):
    st.warning("Waiting for live data…")
    st.stop()

df = pd.read_csv(LOG_CSV)
if df.empty:
    st.info("Log is empty. The first record will arrive on the next tick.")
    st.stop()

# Ensure numeric
for i in range(1, H+1):
    df[f"y_true_{i}"] = pd.to_numeric(df.get(f"y_true_{i}", 0), errors="coerce")
    df[f"y_pred_{i}"] = pd.to_numeric(df.get(f"y_pred_{i}", 0), errors="coerce")
df["latency_ms"] = pd.to_numeric(df.get("latency_ms", 0), errors="coerce")

last = df.iloc[-1]

# Record latency for session average
if not np.isnan(last["latency_ms"]):
    st.session_state.latency_hist.append(float(last["latency_ms"]))

# Optional host averages
if HAVE_PSUTIL:
    st.session_state.cpu_hist.append(psutil.cpu_percent(interval=None))
    st.session_state.ram_hist.append(psutil.virtual_memory().percent)

# --- “today” mask and sums for cost & % error tiles ---
if "wallclock" in df.columns:
    df["_ts"] = pd.to_datetime(df["wallclock"], errors="coerce")
    today_mask = df["_ts"].dt.date == pd.Timestamp.now().date()
else:
    today_mask = np.ones(len(df), dtype=bool)

actual_today = pd.to_numeric(df.loc[today_mask, "y_true_1"], errors="coerce").fillna(0.0)
pred_today   = pd.to_numeric(df.loc[today_mask, "y_pred_1"], errors="coerce").fillna(0.0)
today_actual_sum = float(actual_today.sum())
today_pred_sum   = float(pred_today.sum())

# Header status tiles

colA, colB, colC, colD = st.columns(4)

# Average % error (today) — cumulative % diff
avg_pct_err_today = safe_pct_err(today_actual_sum, today_pred_sum)
err_color, _ = classify_gar(abs(avg_pct_err_today) if not np.isnan(avg_pct_err_today) else np.nan,
                            ERR_OK, ERR_WARN, invert=False)
colA.markdown(status_box("Average % error (today)",
                         f"{avg_pct_err_today:+.1f}%" if not np.isnan(avg_pct_err_today) else "n/a",
                         err_color),
              unsafe_allow_html=True)

# Latency (average)
avg_lat = np.mean(st.session_state.latency_hist) if st.session_state.latency_hist else np.nan
lat_color, _ = classify_gar(avg_lat, LAT_OK, LAT_WARN, invert=False)
colB.markdown(status_box("Latency (average)",
                         f"{avg_lat:.1f} ms" if not np.isnan(avg_lat) else "n/a",
                         lat_color),
              unsafe_allow_html=True)

# CPU usage (average)
if HAVE_PSUTIL and st.session_state.cpu_hist:
    avg_cpu = float(np.mean(st.session_state.cpu_hist))
else:
    avg_cpu = np.nan
cpu_color, _ = classify_gar(avg_cpu, CPU_OK, CPU_WARN, invert=False)
colC.markdown(status_box("CPU usage (average)",
                         f"{avg_cpu:.1f}%" if not np.isnan(avg_cpu) else "n/a",
                         cpu_color),
              unsafe_allow_html=True)

# RAM usage (average)
if HAVE_PSUTIL and st.session_state.ram_hist:
    avg_ram = float(np.mean(st.session_state.ram_hist))
else:
    avg_ram = np.nan
ram_color, _ = classify_gar(avg_ram, RAM_OK, RAM_WARN, invert=False)
colD.markdown(status_box("RAM usage (average)",
                         f"{avg_ram:.1f}%" if not np.isnan(avg_ram) else "n/a",
                         ram_color),
              unsafe_allow_html=True)

st.markdown("---")

# Next 6 hours + Horizon view (with confidence bands)

cA, cB = st.columns([1, 2])

# Prepare latest horizon arrays
y_true = np.array([last.get(f"y_true_{i}", np.nan) for i in range(1, H+1)], dtype=float)
y_pred = np.array([last.get(f"y_pred_{i}", np.nan) for i in range(1, H+1)], dtype=float)

# Human-friendly window label
if "window_start" in df.columns and "window_end" in df.columns:
    start_lbl = fmt_ts(last["window_start"])
    end_lbl   = fmt_ts(last["window_end"])
    win_text  = f"Forecast horizon: next 6 hours ({TZ_LABEL})"
else:
    now = datetime.now()
    win_text = f"Forecast horizon: next 6 hours ({TZ_LABEL})"

with cA:
    st.subheader("Next 6 hours")
    st.caption(win_text)

    # Sum of next 12 half-hour steps
    next_6h_kwh = float(np.nansum(y_pred[:12])) if H >= 12 else float(np.nansum(y_pred))

    # Gauge (6h)
    st.altair_chart(
        altair_gauge(next_6h_kwh, max_value=GAUGE_MAX_6H, threshold=GAUGE_THRESH_6H, title="kWh (next 6h)"),
        use_container_width=False
    )

    # Cost estimator (next 6h + today)
    next_6h_cost     = next_6h_kwh * TARIFF_PER_KWH
    today_pred_cost  = today_pred_sum * TARIFF_PER_KWH
    today_actual_cost= today_actual_sum * TARIFF_PER_KWH

    st.markdown(
        f"""
        <div style="margin-top:8px; font-size:0.95rem;">
          <div><b>Predicted cost (next 6h):</b> €{next_6h_cost:.2f}</div>
          <div><b>Predicted cost (today):</b> €{today_pred_cost:.2f}</div>
          <div><b>Actual cost (today):</b> €{today_actual_cost:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cB:
    st.subheader("Horizon view")

    # Confidence bands from recent history (per-horizon RMSE)
    df_hist = df.tail(CONF_HISTORY_N).copy()

    rmse_per_h = []
    for i in range(1, H+1):
        a = pd.to_numeric(df_hist.get(f"y_true_{i}", np.nan), errors="coerce")
        p = pd.to_numeric(df_hist.get(f"y_pred_{i}", np.nan), errors="coerce")
        m = (~a.isna()) & (~p.isna())
        if m.any():
            err = (p[m] - a[m]).to_numpy(dtype=float)
            rmse_per_h.append(float(np.sqrt(np.mean(err**2))))
        else:
            rmse_per_h.append(np.nan)
    rmse_per_h = np.array(rmse_per_h, dtype=float)
    rmse_fill = np.nanmedian(rmse_per_h) if np.isnan(rmse_per_h).any() else 0.0

    lower = np.maximum(0.0, y_pred - CONF_Z * np.nan_to_num(rmse_per_h, nan=rmse_fill))
    upper = np.maximum(0.0, y_pred + CONF_Z * np.nan_to_num(rmse_per_h, nan=rmse_fill))

    horizon_minutes = np.arange(30, 30*(H+1), 30)  # 30, 60, …, 30*H

    band_df = pd.DataFrame({
        "Minutes ahead": horizon_minutes,
        "Lower": lower,
        "Upper": upper
    })
    lines_df = pd.DataFrame({
        "Minutes ahead": horizon_minutes,
        "Actual": y_true,
        "Predicted": y_pred
    }).melt(id_vars=["Minutes ahead"], var_name="Series", value_name="kWh")

    band = (
        alt.Chart(band_df)
        .mark_area(opacity=0.18)
        .encode(
            x=alt.X("Minutes ahead:Q", title="Minutes ahead"),
            y=alt.Y("Lower:Q", title="kWh"),
            y2="Upper:Q",
            color=alt.value("#60a5fa")
        )
    )
    lines = (
        alt.Chart(lines_df)
        .mark_line()
        .encode(
            x=alt.X("Minutes ahead:Q", title="Minutes ahead"),
            y=alt.Y("kWh:Q", title="kWh"),
            color=alt.Color("Series:N",
                           scale=alt.Scale(domain=["Actual","Predicted"],
                                           range=["#e5e7eb","#93c5fd"]))
        )
    )
    horizon_chart = (
        (band + lines)
        .properties(height=280)
        .configure_axis(labelColor="#e5e7eb", titleColor="#e5e7eb")
        .configure_legend(labelColor="#e5e7eb", titleColor="#e5e7eb")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(horizon_chart, use_container_width=True)

st.markdown("---")

# Today so far (cumulative & table)

st.subheader("Today so far")

colX, colY, colZ = st.columns(3)
colX.metric("Actual (today)", f"{today_actual_sum:.2f} kWh")
colY.metric("Predicted (today)", f"{today_pred_sum:.2f} kWh")
colZ.metric("Difference", f"{(today_pred_sum - today_actual_sum):+.2f} kWh")

roll = pd.DataFrame({
    "Sample": np.arange(len(df)),
    "Actual (cum)": pd.to_numeric(df.get("y_true_1", 0), errors="coerce").fillna(0.0).cumsum(),
    "Predicted (cum)": pd.to_numeric(df.get("y_pred_1", 0), errors="coerce").fillna(0.0).cumsum(),
}).set_index("Sample")
st.line_chart(roll)

st.markdown("---")
st.subheader("Recent records")
st.dataframe(df.tail(12), use_container_width=True)
