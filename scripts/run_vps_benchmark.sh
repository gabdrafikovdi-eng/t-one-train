#!/usr/bin/env bash
# ONE-command T-one CPU performance benchmark on the VPS.
#
#   ./scripts/run_vps_benchmark.sh              # full benchmark (all experiments)
#   ./scripts/run_vps_benchmark.sh --limit 5    # 5-WAV smoke test
#
# What it does:
#   1. builds the benchmark image (Dockerfile.vps-benchmark)
#   2. starts the container (docker-compose.vps-benchmark.yml)
#   3. loads T-one + official KenLM (cached in the hf-model-cache volume)
#   4. runs: cold start -> sequential baseline -> thread sweep -> concurrency
#      sweep -> combined threads+concurrency variants
#   5. writes results to ./results/vps_benchmark_<timestamp>/ on the HOST
#   6. exits cleanly (no manual container access needed)
#
# Equivalent raw command:
#   docker compose -f docker-compose.vps-benchmark.yml up --build --abort-on-container-exit

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required (docker compose plugin v2)." >&2
    exit 1
fi

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export BENCHMARK_ARGS="${BENCHMARK_ARGS:-$*}"
export BENCHMARK_DOCKER_VERSION="$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo unknown)"
export BENCHMARK_IMAGE_ID="$(docker images --no-trunc --format '{{.ID}}' t-one-vps-benchmark:latest 2>/dev/null | head -1 || true)"
[ -n "${BENCHMARK_IMAGE_ID}" ] || export BENCHMARK_IMAGE_ID="$(docker build -q -f Dockerfile.vps-benchmark -t t-one-vps-benchmark:latest . 2>/dev/null | tail -1 || true)"

echo "==> T-one VPS benchmark: image=t-one-vps-benchmark:latest"
echo "==> HOST_UID=${HOST_UID} HOST_GID=${HOST_GID} BENCHMARK_ARGS=${BENCHMARK_ARGS:-<full run>}"

# --abort-on-container-exit: return the container's exit code to the caller
# (CI-friendly; the benchmark itself always finishes on its own).
docker compose -f docker-compose.vps-benchmark.yml up --build \
    --abort-on-container-exit --exit-code-from vps-benchmark
