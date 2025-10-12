set -euo pipefail
IMAGE=prompt-paper:latest
docker build -t "$IMAGE" .
echo "Built $IMAGE"