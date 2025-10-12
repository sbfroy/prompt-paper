set -euo pipefail
IMAGE=prompt-paper:latest
VLLM_NAME=vllm-server
VLLM_PORT=8000
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

# checks if vLLM is running, if not starts it
if ! docker ps --format '{{.Names}}' | grep -q "^${VLLM_NAME}$"; then
  echo "Starting vLLM..."
  docker run -d --name "$VLLM_NAME" --gpus all -p ${VLLM_PORT}:8000 vllm/vllm-openai:latest \
    --model "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 --disable-log-stats
fi

docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -v "$PWD":/workspace -w /workspace \
  "$IMAGE" \
  bash