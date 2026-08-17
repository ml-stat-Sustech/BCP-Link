#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SETTINGS_FILE="${SCRIPT_DIR}/settings.sh"
if [[ ! -r "${SETTINGS_FILE}" ]]; then
  echo "Missing ${SETTINGS_FILE}; run bash setup.sh first." >&2
  exit 2
fi

# shellcheck source=settings.sh
source "${SETTINGS_FILE}"

: "${BCP_LINK_CORPUS:?Set BCP_LINK_CORPUS in scripts/settings.sh}"
: "${ELASTICSEARCH_URL:?Set ELASTICSEARCH_URL in scripts/settings.sh}"
: "${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH in scripts/settings.sh}"
: "${EMBEDDING_BASE_URL:?Set EMBEDDING_BASE_URL in scripts/settings.sh}"

ELASTICSEARCH_URL="${ELASTICSEARCH_URL%/}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL%/}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-a}"
ES_INDEX="browsecomp_plus_link_qwen3_embedding_8b"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v sed >/dev/null 2>&1 || fail "sed is required"
command -v uv >/dev/null 2>&1 || fail "uv is required; run bash setup.sh"
[[ -r "${BCP_LINK_CORPUS}" ]] || fail "BCP-Link corpus is not readable: ${BCP_LINK_CORPUS}"
[[ "${EMBEDDING_BASE_URL}" != *"your-embedding-host"* ]] \
  || fail "Replace EMBEDDING_BASE_URL in scripts/settings.sh"

es_response="$(curl --fail --silent --show-error --max-time 10 "${ELASTICSEARCH_URL}")" \
  || fail "Elasticsearch is not reachable at ${ELASTICSEARCH_URL}"
es_version="$(sed -n 's/.*"number"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  <<<"${es_response}" | head -n 1)"
[[ "${es_version}" == "8.19.18" ]] \
  || fail "Elasticsearch 8.19.18 is required; found ${es_version:-unknown}"

curl --fail --silent --show-error --max-time 35 \
  "${ELASTICSEARCH_URL}/_cluster/health?wait_for_status=yellow&timeout=30s" \
  >/dev/null || fail "Elasticsearch cluster is not ready at ${ELASTICSEARCH_URL}"

api_root="${EMBEDDING_BASE_URL}"
service_root="${EMBEDDING_BASE_URL}"
if [[ "${api_root}" != */v1 ]]; then
  api_root="${api_root}/v1"
else
  service_root="${service_root%/v1}"
fi
curl --fail --silent --show-error --max-time 10 "${service_root}/health" >/dev/null 2>&1 \
  || curl --fail --silent --show-error --max-time 10 \
    -H "Authorization: Bearer ${EMBEDDING_API_KEY}" "${api_root}/models" >/dev/null \
  || fail "Embedding endpoint is not ready at ${EMBEDDING_BASE_URL}"

uv run python -m searcherkit plugins deploy browsecomp-plus-link \
  --dataset_path "${BCP_LINK_CORPUS}" \
  --es_host "${ELASTICSEARCH_URL}" \
  --index_name "${ES_INDEX}" \
  --model_name "${EMBEDDING_MODEL_PATH}" \
  --embedding_url "${EMBEDDING_BASE_URL}" \
  --embedding_api_key "${EMBEDDING_API_KEY}" \
  --embedding_dim 4096 \
  --embedding_max_input_tokens 8192 \
  --prompt_strategy qwen3 \
  --dense-vector \
  --batch_size 100 \
  --embedding_batch_size 16 \
  --max_text_chars 32768 \
  --shards 2 \
  --replicas 0 \
  --overwrite
