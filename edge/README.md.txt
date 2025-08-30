# Edge harness (Docker)
Limits: 1 vCPU, 512 MB RAM, 1 thread, no GPU

## Build
docker build -t edgemeter-tflite .

## Microbenchmark 
docker run --rm --cpus=1 --cpuset-cpus=0 --memory=512m \
  -v "$PWD/models:/models:ro" -v "$PWD/data:/data:ro" -v "$PWD/results:/results" \
  edgemeter-tflite \
  python /app/bench.py --model /models/TinyFormer/TinyFormer_fp16.tflite \
  --x_path /data/X_test_48_12.npy --warmup 50 --runs 500 --threads 1 --out /results


