#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SETTINGS_FILE="${SCRIPT_DIR}/settings.sh"
if [[ ! -r "${SETTINGS_FILE}" ]]; then
  echo "Missing ${SETTINGS_FILE}; run bash setup.sh or copy scripts/settings.example.sh." >&2
  exit 2
fi

# shellcheck source=settings.sh
source "${SETTINGS_FILE}"

: "${JUDGE_MODEL:?Set JUDGE_MODEL in scripts/settings.sh}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL in scripts/settings.sh}"
: "${JUDGE_API_KEY:?Set JUDGE_API_KEY in scripts/settings.sh}"

if [[ "${JUDGE_MODEL}" == "your-judge-model" || "${JUDGE_BASE_URL}" == *"your-judge-host"* ]]; then
  echo "Replace the judge placeholders in scripts/settings.sh." >&2
  exit 2
fi

GENERATION_DIR="${GENERATION_DIR:-${RUN_DIR}/generation}"
HISTORY_DIR="${HISTORY_DIR:-${GENERATION_DIR}/history}"
EVALUATION_DIR="${EVALUATION_DIR:-${RUN_DIR}/evaluation}"

args=(
  uv run python -m searcherkit evaluate
  "${HISTORY_DIR}"
  "${EVALUATION_DIR}"
  --judge-model "${JUDGE_MODEL}"
  --judge-base-url "${JUDGE_BASE_URL}"
  --judge-api-key "${JUDGE_API_KEY}"
  --max-concurrency "${JUDGE_MAX_CONCURRENCY}"
)

if [[ -n "${BENCHMARK_TOTAL}" ]]; then
  args+=(--benchmark-total "${BENCHMARK_TOTAL}")
fi

"${args[@]}"
