#!/bin/bash

source .env

API_KEY="${LLM_API_KEY}"
PORT="${LLM_PORT}"
MODEL_NAME="${LLM_MODEL}"
GPUS="${LLM_GPUS}"


CUDA_VISIBLE_DEVICES=$GPUS vllm serve \
    --tensor-parallel-size $(( $(echo "$GPUS" | awk -F',' '{print NF}') )) \
    --api-key "${API_KEY}" \
    --dtype auto \
    --port "${PORT}" \
    "${MODEL_NAME}"