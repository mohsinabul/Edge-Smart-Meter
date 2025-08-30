# edge/Dockerfile  — minimal, portable, CPU-only image for TFLite tests
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

# Small deps for psutil & stability; tini for clean shutdown
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 procps tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY bench.py edge_mimic.py /app/

# Prefer tflite-runtime (small). If unavailable for your arch, fall back to TF CPU
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 psutil==5.9.8 \
 && (pip install --no-cache-dir ai-edge-litert || pip install --no-cache-dir tflite-runtime==2.14.0)

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python","-c","print('Edge image ready. Mount models/data and run bench.py or edge_mimic.py')"]
