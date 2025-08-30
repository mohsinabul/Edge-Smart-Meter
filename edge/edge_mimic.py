# edge_mimic.py
import os, time, json, argparse, platform
import numpy as np
import psutil

# --- Interpreter import: ai-edge-litert -> tflite-runtime -> tensorflow ---
try:
    from ai_edge_litert.interpreter import Interpreter        # LiteRT 
    RUNTIME_NAME = "LiteRT"
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter     # TFLite runtime (fallback)
        RUNTIME_NAME = "tflite-runtime"
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter  # TF full (last resort)
        RUNTIME_NAME = "tf.lite (TensorFlow)"

def load_itp(model_path: str, threads: int):
    try:
        itp = Interpreter(model_path=model_path, num_threads=int(threads))
    except TypeError:
        itp = Interpreter(model_path=model_path)
    itp.allocate_tensors()
    return itp

def load_stream(path: str):
    arr = np.load(path, allow_pickle=False)
    # Ensure [N, T, F] -> [N, T, F] 
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def percentile(a, q):
    if not a: 
        return float("nan")
    return float(np.percentile(a, q))

def mimic_loop(itp, stream, tick_seconds: float, limit: int, log_series: bool):
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]

    # Determine input dims
    want_shape = list(inp["shape"])  # e.g., [1, 48, F]
    T = stream.shape[1] if stream.ndim == 3 else (want_shape[1] if len(want_shape) > 1 else 48)
    F = stream.shape[2] if stream.ndim == 3 else (want_shape[2] if len(want_shape) > 2 else 1)

    lat_ms = []
    rss_mb_series = []
    cpu_pct_series = []
    t_rel_series = []

    p = psutil.Process(os.getpid())
    cpu_prev = sum(p.cpu_times()[:2])  # user+sys
    wall_prev = time.perf_counter()

    n = min(limit, stream.shape[0]) if limit else stream.shape[0]
    t0 = time.perf_counter()

    for i in range(n):
        x = stream[i]
        # Ensure shape [1, T, F]
        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        elif x.ndim == 3 and x.shape[0] != 1:
            x = x[:1]  # take first batch window if pre-batched
        x = x.astype(np.float32, copy=False)

        t1 = time.perf_counter()
        itp.set_tensor(inp["index"], x)
        itp.invoke()
        _ = itp.get_tensor(out["index"])
        t2 = time.perf_counter()

        lat = (t2 - t1) * 1000.0
        lat_ms.append(lat)

        # Memory & approximate CPU util since last tick
        rss_mb = p.memory_info().rss / 1e6
        cpu_now = sum(p.cpu_times()[:2])
        wall_now = time.perf_counter()
        cpu_pct = 0.0
        if wall_now > wall_prev:
            cpu_pct = (cpu_now - cpu_prev) / (wall_now - wall_prev) * 100.0
        cpu_prev, wall_prev = cpu_now, wall_now

        if log_series:
            rss_mb_series.append(rss_mb)
            cpu_pct_series.append(cpu_pct)
            t_rel_series.append(wall_now - t0)

        # Keep the feed interval roughly constant
        sleep_s = max(0.0, tick_seconds - (t2 - t1))
        time.sleep(sleep_s)

    # Summary
    rss_mb_max = max(rss_mb_series) if rss_mb_series else p.memory_info().rss / 1e6
    tick_ms = tick_seconds * 1000.0

    # Overrun & headroom metrics
    overruns = sum(1 for v in lat_ms if v > tick_ms) if tick_ms > 0 else 0
    overrun_rate = (overruns / n) if n else 0.0
    lat_mean = float(np.mean(lat_ms)) if lat_ms else float("nan")
    lat_p50 = percentile(lat_ms, 50)
    lat_p95 = percentile(lat_ms, 95)
    lat_p99 = percentile(lat_ms, 99)
    jitter_ms = (lat_p95 - lat_p50) if np.isfinite(lat_p95) and np.isfinite(lat_p50) else float("nan")

    headroom_mean = (1.0 - (lat_mean / tick_ms)) if tick_ms > 0 and np.isfinite(lat_mean) else None
    headroom_p95  = (1.0 - (lat_p95  / tick_ms)) if tick_ms > 0 and np.isfinite(lat_p95)  else None

    # RSS percentiles if we logged the series
    rss_p50 = percentile(rss_mb_series, 50) if rss_mb_series else None
    rss_p95 = percentile(rss_mb_series, 95) if rss_mb_series else None

    summary = {
        "runs": n,
        "tick_seconds": tick_seconds,

        "lat_ms_mean": lat_mean,
        "lat_ms_p50": lat_p50,
        "lat_ms_p95": lat_p95,
        "lat_ms_p99": lat_p99,           
        "jitter_ms_p95_p50": jitter_ms,  
        "lat_ms_min": float(np.min(lat_ms)) if lat_ms else float("nan"),
        "lat_ms_max": float(np.max(lat_ms)) if lat_ms else float("nan"),

        "overrun_rate": overrun_rate,    
        "headroom_mean": headroom_mean,  
        "headroom_p95": headroom_p95,    

        "rss_mb_max": float(rss_mb_max),
        "rss_mb_p50": rss_p50,           
        "rss_mb_p95": rss_p95,           

        "cpu_util_pct_appx": float(np.mean(cpu_pct_series)) if cpu_pct_series else None,
        "runtime_name": RUNTIME_NAME,    
        "ts": int(time.time()),
    }

    series = None
    if log_series:
        series = {
            "time_s": t_rel_series,
            "lat_ms": lat_ms,
            "rss_mb": rss_mb_series,
            "cpu_util_pct_appx": cpu_pct_series,
        }
    return summary, series

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stream_path", required=True)
    ap.add_argument("--tick_seconds", type=float, default=1.5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = full stream")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--log_series", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    itp = load_itp(args.model, args.threads)
    stream = load_stream(args.stream_path)

    summary, series = mimic_loop(
        itp, stream, tick_seconds=args.tick_seconds,
        limit=args.limit, log_series=args.log_series
    )

    # Attach identifiers and saving
    try:
        model_size_mb = round(os.path.getsize(args.model) / 1_000_000, 3)
    except Exception:
        model_size_mb = None

    summary.update({
        "model_path": args.model,
        "stream_path": args.stream_path,
        "threads": args.threads,
        "out_dir": args.out_dir,
        "model_size_mb": model_size_mb,   
    })

    with open(os.path.join(args.out_dir, "mimic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if series:
        # Writing CSV for easy plotting
        import pandas as pd
        df = pd.DataFrame(series)
        df.to_csv(os.path.join(args.out_dir, "mimic_series.csv"), index=False)

    print("\nMIMIC SUMMARY")
    for k, v in summary.items():
        print(f"{k:>18}: {v}")

if __name__ == "__main__":
    main()
