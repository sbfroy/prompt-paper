#!/bin/bash

source /workspace/.env

API_KEY="${EMBEDD_API_KEY}"
PORT="${EMBEDD_PORT}"
MODEL_NAME="${EMBEDD_MODEL}"
GPUS="${EMBEDD_GPUS}"

echo "Starting vLLM embedding server with the following configurations:"
echo "Port: ${PORT}"
echo "Model Name: ${MODEL_NAME}"
echo "GPUs: ${GPUS}"

CUDA_VISIBLE_DEVICES=$GPUS vllm serve \
    --tensor-parallel-size $(( $(echo "$GPUS" | awk -F',' '{print NF}') )) \
    --api-key "${API_KEY}" \
    --dtype auto \
    --port "${PORT}" \
    --task embedding \
    "${MODEL_NAME}"