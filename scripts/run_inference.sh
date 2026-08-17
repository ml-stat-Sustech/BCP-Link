#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SETTINGS_FILE="${SCRIPT_DIR}/settings.sh"
if [[ ! -r "${SETTINGS_FILE}" ]]; then
  echo "Missing ${SETTINGS_FILE}; create it and fill in the required values." >&2
  exit 2
fi

# shellcheck source=settings.sh
source "${SETTINGS_FILE}"

while (( $# > 0 )); do
  if (( $# < 2 )); then
    echo "Missing value for $1" >&2
    exit 2
  fi
  case "$1" in
    --model-type) MODEL_TYPE="$2" ;;
    --model) MODEL_NAME="$2" ;;
    --base-url) LLM_BASE_URLS="[\"$2\"]" ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift 2
done

: "${MODEL_TYPE:?Set MODEL_TYPE in scripts/settings.sh or pass --model-type}"
: "${MODEL_NAME:?Set MODEL_NAME in scripts/settings.sh or pass --model}"
: "${LLM_BASE_URLS:?Set LLM_BASE_URLS in scripts/settings.sh or pass --base-url}"
: "${LLM_API_KEY:?Set LLM_API_KEY in scripts/settings.sh}"
: "${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH in scripts/settings.sh}"
: "${EMBEDDING_BASE_URL:?Set EMBEDDING_BASE_URL in scripts/settings.sh}"
: "${EMBEDDING_API_KEY:?Set EMBEDDING_API_KEY in scripts/settings.sh}"
: "${ELASTICSEARCH_URL:?Set ELASTICSEARCH_URL in scripts/settings.sh}"
: "${BCP_LINK_DATASET:?Set BCP_LINK_DATASET in scripts/settings.sh}"

if [[ "${LLM_BASE_URLS}" != \[*\] ]]; then
  echo "LLM_BASE_URLS must be a JSON-style list, got: ${LLM_BASE_URLS}" >&2
  exit 2
fi
if [[ "${LLM_BASE_URLS}" == *"your-llm-host"* ]]; then
  echo "Replace the LLM endpoint placeholder in scripts/settings.sh." >&2
  exit 2
fi

case "${MODEL_TYPE}" in
  local) CONFIG_PATH="searcherkit" ;;
  closed) CONFIG_PATH="searcherkit_closed" ;;
  *)
    echo "MODEL_TYPE must be local or closed, got: ${MODEL_TYPE}" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_NAME}"
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-131072}"
RUN_DIR="${RUN_DIR:-outputs/bcp-link}"
GENERATION_DIR="${GENERATION_DIR:-${RUN_DIR}/generation}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
LLM_CONCURRENCY_LIMIT="${LLM_CONCURRENCY_LIMIT:-${MAX_CONCURRENCY}}"
ES_MAX_CONCURRENCY="${ES_MAX_CONCURRENCY:-${MAX_CONCURRENCY}}"
EMBEDDING_MAX_CONCURRENCY="${EMBEDDING_MAX_CONCURRENCY:-${MAX_CONCURRENCY}}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"
CHECKPOINT_RESUME="${CHECKPOINT_RESUME:-true}"
CONTENT_FIELD="${CONTENT_FIELD:-text}"

export MODEL_PATH
export LLM_BASE_URLS
export LLM_API_KEY
export EMBEDDING_MODEL_PATH
export EMBEDDING_BASE_URL
export EMBEDDING_API_KEY
export ELASTICSEARCH_URL
export BCP_LINK_DATASET

args=(
  uv run python -m searcherkit run
  --config-path "${CONFIG_PATH}"
  "agent.max_tokens=${MODEL_CONTEXT_LENGTH}"
  "agent.llm_client.base_url=${LLM_BASE_URLS}"
  "agent.llm_client.concurrency_limit=${LLM_CONCURRENCY_LIMIT}"
  "agent.sources.0.es_max_concurrency=${ES_MAX_CONCURRENCY}"
  "agent.sources.0.embedding_max_concurrency=${EMBEDDING_MAX_CONCURRENCY}"
  "agent.sources.0.search_fields=[title,${CONTENT_FIELD}]"
  "agent.sources.0.text_field=${CONTENT_FIELD}"
  "max_concurrency=${MAX_CONCURRENCY}"
  "output_path=${GENERATION_DIR}"
  "overwrite_output=${OVERWRITE_OUTPUT}"
  "checkpoint.resume=${CHECKPOINT_RESUME}"
)

if [[ -n "${MAX_ITEMS:-}" ]]; then
  args+=("dataloader.max_items=${MAX_ITEMS}")
fi

"${args[@]}"
