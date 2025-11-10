set -euo pipefail
IMAGE=grasp:latest
docker build -t "$IMAGE" .
echo "Built $IMAGE"