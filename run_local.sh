set -euo pipefail # Exit on error (safety setting)
IMAGE=prompt-paper:latest

docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -v "$PWD":/workspace -w /workspace \
  "$IMAGE" \
  bash