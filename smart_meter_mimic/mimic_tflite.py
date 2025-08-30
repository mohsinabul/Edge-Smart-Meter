# mimic_tflite.py
import os, sys, time, json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# TFLite interpreter selection
Interpreter = None
try:
    # Prefer lightweight tflite_runtime if available
    from tflite_runtime.interpreter import Interpreter  
except Exception:
    try:
        # Fall back to TF’s built-in lite interpreter
        from tensorflow.lite import Interpreter  
    except Exception as e:
        print("[fatal] No TFLite interpreter available. Install tflite-runtime or tensorflow.", file=sys.stderr)
        raise

# Env + defaults

DATA_DIR     = Path(os.getenv("DATA_DIR", "/app/data"))
LOG_DIR      = Path(os.getenv("LOG_DIR", "/app/logs"))
LOG_CSV      = Path(os.getenv("LOG_CSV", str(LOG_DIR / "live_predictions.csv")))
STATE_JSON   = Path(os.getenv("STATE_JSON", str(LOG_DIR / "offset.json")))
TFLITE_PATH  = Path(os.getenv("TFLITE_PATH", str(Path("/app/model") / "Final_LiPFormer_Model_48_12_int8.tflite")))

X_PATH       = Path(os.getenv("X_PATH", str(DATA_DIR / "X_demo_48_12.npy")))
Y_PATH       = Path(os.getenv("Y_PATH", str(DATA_DIR / "y_demo_48_12.npy")))
H            = int(os.getenv("HORIZON", "12"))
SLEEP_SEC    = int(os.getenv("SLEEP_SECONDS", "1800"))  # 1800 = 30 minutes
TZ_LABEL     = os.getenv("TIMEZONE", "Europe/Dublin")


# Helpers: state & logging

def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def read_state(state_path: Path) -> int | None:
    try:
        if not state_path.exists():
            return None
        with state_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return int(d.get("last_index"))
    except Exception:
        return None

def write_state(state_path: Path, idx: int) -> None:
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"last_index": int(idx)}, f)
    tmp.replace(state_path)

def csv_has_header(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            first = f.readline()
        return "index,wallclock" in first.replace(" ", "").lower()
    except Exception:
        return False

def write_header_if_needed(csv_path: Path, horizon: int):
    if csv_path.exists() and csv_has_header(csv_path):
        return
    cols = ["index", "wallclock", "window_start", "window_end", "latency_ms"]
    for i in range(1, horizon + 1):
        cols.append(f"y_true_{i}")
    for i in range(1, horizon + 1):
        cols.append(f"y_pred_{i}")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

def append_row(csv_path: Path, row_dict: dict):
    line = ",".join(str(row_dict[k]) for k in row_dict.keys())
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# TFLite runner

class TFLiteRunner:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        self.interp = Interpreter(model_path=str(model_path), num_threads=int(os.getenv("NUM_THREADS", "1")))
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x shape: (1, 48, F). Returns (1, H) or (H,) depending on model."""
        # some tflite models expect float32 input
        x = x.astype(np.float32, copy=False)
        self.interp.set_tensor(self.input_details[0]["index"], x)
        t0 = time.perf_counter()
        self.interp.invoke()
        t1 = time.perf_counter()
        out = self.interp.get_tensor(self.output_details[0]["index"])
        # Normalize shape to (H,)
        out = np.array(out)
        if out.ndim == 2 and out.shape[0] == 1:
            out = out[0]
        return out, (t1 - t0) * 1000.0  # ms


# Main

def main():
    ensure_dirs()
    tz = ZoneInfo(TZ_LABEL)

    print(f"[mimic:tflite] Loading TFLite model: {TFLITE_PATH}")
    runner = TFLiteRunner(TFLITE_PATH)

    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(f"Missing arrays. Expected:\n  {X_PATH}\n  {Y_PATH}")

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    # Expect X: (N, 48, F), Y: (N, H)
    if X.ndim != 3:
        raise ValueError(f"X has wrong shape {X.shape}. Expected (N,48,F).")
    if Y.ndim == 1:
        Y = Y.reshape(-1, H)
    elif Y.shape[1] != H:
        # if y is (N,) assume single-step and tile just for logging
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[1] != H:
            # pad/trim to H for consistent logging (best-effort)
            if Y.shape[1] > H:
                Y = Y[:, :H]
            else:
                Y = np.hstack([Y, np.tile(Y[:, -1:], (1, H - Y.shape[1]))])

    N, T, F = X.shape
    print(f"[mimic:tflite] Loaded X:{X.shape} Y:{Y.shape} (N={N}, T={T}, F={F}, H={H})")
    print(f"[mimic:tflite] Start cadence={SLEEP_SEC}s | tz={TZ_LABEL}")

    # Prepare CSV header
    write_header_if_needed(LOG_CSV, H)

    # Determine starting index from state
    last_idx = read_state(STATE_JSON)
    if last_idx is None:
        start_idx = 0
        print("[mimic:tflite] No prior state found. Starting at idx=0")
    else:
        start_idx = (last_idx + 1) % N
        print(f"[mimic:tflite] Resuming from last_index={last_idx} -> start_idx={start_idx}")

    # Cadence alignment with low drift
    next_tick = time.monotonic()

    idx = start_idx
    while True:
        # Prepare input window and timestamp labels
        x1 = X[idx:idx + 1]  # (1, 48, F)
        # wallclock now, plus a 30-min horizon label window
        now = datetime.now(tz)
        window_start = now.strftime("%Y-%m-%dT%H:%M:%S%z")
        window_end = (now + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S%z")

        # Ground truth (best-effort: take Y[idx], fill NaNs if needed)
        y_true = Y[idx]
        if y_true is None or (isinstance(y_true, float) and np.isnan(y_true)):
            y_true = np.zeros(H, dtype=np.float32)

        # Predict
        y_pred, latency_ms = runner.predict(x1)

        # Build row
        row = {
            "index": idx,
            "wallclock": now.isoformat(),
            "window_start": window_start,
            "window_end": window_end,
            "latency_ms": f"{latency_ms:.3f}",
        }
        for i in range(H):
            row[f"y_true_{i+1}"] = float(y_true[i]) if i < len(y_true) else ""
        for i in range(H):
            row[f"y_pred_{i+1}"] = float(y_pred[i]) if i < len(y_pred) else ""

        append_row(LOG_CSV, row)
        write_state(STATE_JSON, idx)

        # Progress
        print(f"[mimic:tflite] {now.isoformat()} | idx={idx} | latency={latency_ms:.1f} ms")

        # Advance index (circular, live forever)
        idx = (idx + 1) % N

        # Sleep with drift control
        next_tick += SLEEP_SEC
        sleep_for = max(0.0, next_tick - time.monotonic())
        time.sleep(sleep_for)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[mimic:tflite] Stopped by user.")
