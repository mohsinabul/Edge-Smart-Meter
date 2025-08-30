# -*- coding: utf-8 -*-
"""
EdgeMeter Docker runner: build image, smoke test, full microbench, and edge-mimic.

Usage:
  - Open in Spyder (or run with Python).
  - Scroll to the __main__ block and UNCOMMENT exactly ONE action at a time.

Assumes the repo layout:
  edge/
    Dockerfile
    bench.py
    edge_mimic.py
    models/<Model>/<Model>_<variant>.tflite
    data/X_test_48_12.npy
    data/streams/edge_stream_{low,med,high}.npy
    results/   (created if missing)

Author: abul mohsin
Date:   2025-08-14
"""
import os, sys, subprocess, shutil, pandas as pd
import glob, json, re
import time 

# Config

IMAGE = "edgemeter-tflite"

HOST_ROOT = os.path.abspath(os.getcwd())  # .../edge
MODELS    = os.path.join(HOST_ROOT, "models")
DATA      = os.path.join(HOST_ROOT, "data")
RESULTS   = os.path.join(HOST_ROOT, "results")

# Preferred candidates for the smoke test (first found wins)
CANDIDATES = [
    ("TinyFormer", "TinyFormer_fp16.tflite"),
    ("TinyFormer", "TinyFormer_int8_dynamic.tflite"),
    ("LSTM",       "LSTM_fp16.tflite"),
    ("LiteFormer", "LiteFormer_fp16.tflite"),
    ("LiPFormer",  "LiPFormer_fp16.tflite"),
]

# Matrix for full bench / mimic
MODEL_DIRS = {
    "LSTM":       "LSTM",
    "TinyFormer": "TinyFormer",
    "LiPFormer":  "LiPFormer",
    "LiteFormer": "LiteFormer",
}
VARIANTS = ["fp32", "fp16", "int8_dynamic", "int8_full"]
STREAMS  = ["low", "med", "high"]

# Helpers

def run(cmd:list, check=True):
    """Run a shell command with UTF-8 output (Spyder/Windows safe)."""
    print(">>", " ".join(cmd))
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.stdout:
        print(p.stdout)
    if check and p.returncode != 0:
        raise SystemExit(f"Command failed ({p.returncode})")
    return p

def ensure_paths(require_streams=False):
    assert shutil.which("docker"), "Docker not found. Install Docker Desktop / docker engine."
    for pth in (MODELS, DATA):
        assert os.path.exists(pth), f"Missing path: {pth}"
    assert os.path.exists(os.path.join(DATA, "X_test_48_12.npy")), "Missing data/X_test_48_12.npy"
    if require_streams:
        for s in STREAMS:
            sp = os.path.join(DATA, "streams", f"edge_stream_{s}.npy")
            assert os.path.exists(sp), f"Missing stream file: {sp}"
    os.makedirs(RESULTS, exist_ok=True)

def pick_smoke_model() -> str:
    """Return container path '/models/...tflite' for the smoke test."""
    # First pass: preferred list
    for sub, fname in CANDIDATES:
        host_path = os.path.join(MODELS, sub, fname)
        if os.path.exists(host_path):
            print(f"[smoke] Using model: {host_path}")
            return f"/models/{sub}/{fname}"
    # Fallback: first .tflite anywhere under models/
    for root, _, files in os.walk(MODELS):
        for f in files:
            if f.endswith(".tflite"):
                rel = os.path.relpath(os.path.join(root, f), MODELS).replace("\\", "/")
                print(f"[smoke] Fallback model: {os.path.join(MODELS, rel)}")
                return f"/models/{rel}"
    raise SystemExit("No .tflite models found under edge/models/. Please copy your exports there.")

def docker_base():
    """Common docker run flags approximating a constrained smart meter."""
    return [
        "docker","run","--rm",
        "--cpus=1","--cpuset-cpus=0",
        "--memory=512m","--memory-swap=512m",
        "--pids-limit=256",
        "--read-only","--tmpfs","/tmp:rw,noexec,nosuid,size=64m",
        "-e","PYTHONDONTWRITEBYTECODE=1",
        "-e","OMP_NUM_THREADS=1",
        "-e","OPENBLAS_NUM_THREADS=1",
        "-v", f"{MODELS}:/models:ro",
        "-v", f"{DATA}:/data:ro",
        "-v", f"{RESULTS}:/results",
        IMAGE
    ]

# Actions

def build_image():
    """
    Build the image defined by Dockerfile.

    Dockerfile should install:
      - numpy, pandas, psutil
      - ai-edge-litert  (primary)
      - fallback to tflite-runtime==2.14.0 if litert unavailable
    """
    run(["docker","build","--progress=plain","-t", IMAGE, "."])

def smoke_test():
    """Short microbenchmark on one model to verify everything end-to-end."""
    ensure_paths(require_streams=False)
    model_in_container = pick_smoke_model()
    cmd = docker_base() + [
        "python","-B","/app/bench.py",
        "--model",  model_in_container,
        "--x_path", "/data/X_test_48_12.npy",
        "--subset","5000",   
        "--warmup","20",
        "--runs","100",
        "--threads","1",
        "--out","/results",
        "--tag","smoke",
    ]
    run(cmd)

    # Print the newest bench_*.csv to console
    csvs = [f for f in os.listdir(RESULTS) if f.startswith("bench_") and f.endswith(".csv")]
    if not csvs:
        print("No bench_*.csv found in results/")
        return
    newest = max(csvs, key=lambda x: os.path.getmtime(os.path.join(RESULTS, x)))
    df = pd.read_csv(os.path.join(RESULTS, newest))
    print("\nSMOKE RESULT (latest bench CSV):\n", df.to_string(index=False))

def bench_all():
    """Run microbench for every model+variant present on disk."""
    ensure_paths(require_streams=False)
    for mdl, sub in MODEL_DIRS.items():
        for var in VARIANTS:
            host_model = os.path.join(MODELS, sub, f"{mdl}_{var}.tflite")
            if not os.path.exists(host_model):
                print(f"[skip] {host_model} not found")
                continue
            model_in_container = f"/models/{sub}/{mdl}_{var}.tflite"
            tag = f"bench_{mdl}_{var}"
            print(f"\n=== BENCH: {mdl} / {var} ===")
            cmd = docker_base() + [
                "python","-B","/app/bench.py",
                "--model",  model_in_container,
                "--x_path", "/data/X_test_48_12.npy",
                "--subset","5000",
                "--warmup","50",
                "--runs","500",
                "--threads","1",
                "--out","/results",
                "--tag", tag,
            ]
            run(cmd)

def summarize_bench():
    """Aggregate all bench_*.json/CSV into one table and save summary CSV."""
    rows = []
    # Prefer JSON (stable types). Fallback to CSV if JSON missing.
    json_files = sorted(glob.glob(os.path.join(RESULTS, "bench_*.json")))
    csv_files  = sorted(glob.glob(os.path.join(RESULTS, "bench_*.csv"))) if not json_files else []

    def parse_model_variant(model_path: str):
        # model_path 
        base = os.path.basename(model_path)                # TinyFormer_fp16.tflite
        stem = re.sub(r"\.tflite$", "", base)              # TinyFormer_fp16
        parts = stem.split("_")
        model   = parts[0]
        variant = "_".join(parts[1:]) if len(parts) > 1 else "fp32"
        return model, variant

    if json_files:
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            model, variant = parse_model_variant(d["model_path"])
            rows.append({
                "model": model,
                "variant": variant,
                "lat_ms_mean": d["lat_ms_mean"],
                "lat_ms_p50": d["lat_ms_p50"],
                "lat_ms_p95": d["lat_ms_p95"],
                "lat_ms_p99": d.get("lat_ms_p99"),                 
                "jitter_ms_p95_p50": d.get("jitter_ms_p95_p50"),   
                "rss_mb_max": d["rss_mb_max"],
                "cpu_util_pct_appx": d.get("cpu_util_pct_appx", None),
                "runtime_name": d.get("runtime_name"),             
                "model_size_mb": d.get("model_size_mb"),           
                "tag": d.get("tag", ""),
                "ts": d.get("ts", ""),
            })
    else:
        import pandas as pd
        for cf in csv_files:
            df = pd.read_csv(cf)
            if df.empty:
                continue
            d = df.iloc[0].to_dict()
            model, variant = parse_model_variant(d["model_path"])
            rows.append({
                "model": model,
                "variant": variant,
                "lat_ms_mean": d["lat_ms_mean"],
                "lat_ms_p50": d["lat_ms_p50"],
                "lat_ms_p95": d["lat_ms_p95"],
                "lat_ms_p99": d.get("lat_ms_p99"),                 
                "jitter_ms_p95_p50": d.get("jitter_ms_p95_p50"),   
                "rss_mb_max": d["rss_mb_max"],
                "cpu_util_pct_appx": d.get("cpu_util_pct_appx", None),
                "runtime_name": d.get("runtime_name"),             
                "model_size_mb": d.get("model_size_mb"),           
                "tag": d.get("tag", ""),
                "ts": d.get("ts", ""),
            })

    if not rows:
        print("No bench artifacts found.")
        return

    import pandas as pd, subprocess
    summary = pd.DataFrame(rows)
    summary.sort_values(["lat_ms_mean","model","variant"], inplace=True)

    # Attach container image size once (MB) using IMAGE if defined
    image_name = globals().get("IMAGE", "edgemeter-tflite")
    try:
        p = subprocess.run(
            ["docker","image","inspect", image_name, "--format","{{.Size}}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        size_str = (p.stdout or "").strip()
        size_bytes = int(size_str) if (p.returncode == 0 and size_str.isdigit()) else None
        image_mb = round(size_bytes/1_000_000, 1) if size_bytes else None
    except Exception:
        image_mb = None
    summary["container_image_mb"] = image_mb

    out_csv = os.path.join(RESULTS, "bench_summary.csv")
    summary.to_csv(out_csv, index=False)

    # Choose display columns dynamically
    desired = [
        "model","variant",
        "lat_ms_mean","lat_ms_p50","lat_ms_p95","lat_ms_p99","jitter_ms_p95_p50",
        "rss_mb_max","cpu_util_pct_appx","runtime_name","model_size_mb","container_image_mb"
    ]
    cols = [c for c in desired if c in summary.columns]

    print("\n=== BENCH SUMMARY (fastest first) ===")
    print(summary[cols].to_string(index=False))
    print(f"\nSaved: {out_csv}")



def summarize_mimic():
    """Aggregate all mimic_* run summaries into one CSV table."""
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "mimic_*"))):
        if not os.path.isdir(d):
            continue

        # Find a summary JSON
        cand = None
        for name in ("summary.json", "mimic_summary.json"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                cand = p
                break
        if cand is None:
            jsfiles = sorted(glob.glob(os.path.join(d, "*.json")))
            cand = jsfiles[0] if jsfiles else None
        if cand is None:
            print(f"[skip] no summary JSON in {d}")
            continue

        with open(cand, "r", encoding="utf-8") as f:
            s = json.load(f)

        # Parse model / variant / stream from folder name: mimic_<Model>_<variant>_<stream>
        base = os.path.basename(d)
        m = re.match(r"mimic_([^_]+)_(.+)_(low|med|high)$", base)
        model, variant, stream = ("?", "?", "?")
        if m:
            model, variant, stream = m.group(1), m.group(2), m.group(3)

        cpu_val = s.get("cpu_util_pct_appx")
        # If value looks like a fraction (<= 1), convert to percent; else leave as-is.
        if isinstance(cpu_val, (int, float)) and cpu_val is not None and cpu_val <= 1:
            cpu_pct = cpu_val * 100.0
        else:
            cpu_pct = cpu_val

        rows.append({
            "model": model,
            "variant": variant,
            "stream": stream,
            "runs": s.get("runs"),
            "tick_seconds": s.get("tick_seconds"),
            "lat_ms_mean": s.get("lat_ms_mean"),
            "lat_ms_p50": s.get("lat_ms_p50"),
            "lat_ms_p95": s.get("lat_ms_p95"),
            "lat_ms_min": s.get("lat_ms_min"),
            "lat_ms_max": s.get("lat_ms_max"),
            "rss_mb_max": s.get("rss_mb_max"),
            "cpu_util_pct_appx": cpu_pct,
            "ts": s.get("ts"),
            "model_path": s.get("model_path"),
            "stream_path": s.get("stream_path"),
        })

    if not rows:
        print("No mimic artifacts found.")
        return

    df = pd.DataFrame(rows).sort_values(["model", "variant", "stream"])
    out_csv = os.path.join(RESULTS, "mimic_summary.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== MIMIC SUMMARY ===")
    print(df[[
        "model","variant","stream","runs","tick_seconds",
        "lat_ms_mean","lat_ms_p95","rss_mb_max","cpu_util_pct_appx"
    ]].to_string(index=False))
    print(f"\nSaved: {out_csv}")


def mimic_all(tick_seconds=1.5, limit=2000, log_series=True):
    """Edge mimic over low/med/high for each model+variant, with per-run timing."""
    ensure_paths(require_streams=True)
    total_elapsed = 0.0
    n_runs = 0

    for mdl, sub in MODEL_DIRS.items():
        for var in VARIANTS:
            host_model = os.path.join(MODELS, sub, f"{mdl}_{var}.tflite")
            if not os.path.exists(host_model):
                print(f"[skip] {host_model} not found")
                continue

            for stream in STREAMS:
                model_path  = f"/models/{sub}/{mdl}_{var}.tflite"
                stream_path = f"/data/streams/edge_stream_{stream}.npy"
                out_dir     = f"/results/mimic_{mdl}_{var}_{stream}"

                print(f"\n=== MIMIC: {mdl} / {var} / {stream} ===")
                cmd = docker_base() + [
                    "python","-B","/app/edge_mimic.py",
                    "--model", model_path,
                    "--stream_path", stream_path,
                    "--tick_seconds", str(tick_seconds),
                    "--threads","1",
                    "--limit", str(limit),
                    "--out_dir", out_dir
                ]
                if log_series:
                    cmd.append("--log_series")

                # --- timing wraps the container run ---
                t0 = time.time()
                run(cmd)
                elapsed = time.time() - t0
                # -------------------------------------

                print(f"[mimic] {mdl} / {var} / {stream} took {elapsed:.1f}s")
                total_elapsed += elapsed
                n_runs += 1

    if n_runs:
        print(f"\n[mimic] Completed {n_runs} runs in {total_elapsed/60:.1f} minutes "
              f"(avg {total_elapsed/n_runs:.1f}s per run)")

# Entry

if __name__ == "__main__":
    # Uncomment EXACTLY ONE of the following blocks each time you run.

    # 1) Build image only
    # build_image(); sys.exit(0)

    # 2) Build then smoke test a single model
    # build_image(); smoke_test(); sys.exit(0)

    # 3) Build then microbench ALL model+variant files that exist
    # build_image(); bench_all();summarize_bench(); sys.exit(0)

    # 4) Build then run edge-mimic across all (low/med/high)
    # build_image(); bench_all(); summarize_bench(); sys.exit(0)
    build_image(); mimic_all(tick_seconds=1.0, limit=336, log_series=True); summarize_mimic(); sys.exit(0)

