#!/usr/bin/env bash

# Copy this file to scripts/settings.sh and replace the placeholders below.
# scripts/settings.sh is ignored by Git to avoid publishing credentials.
# Existing environment variables take precedence over the defaults in this file.

# Index and retrieval settings.
BCP_LINK_CORPUS="${BCP_LINK_CORPUS:-data/bcp_link_corpus.jsonl}"
ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-http://127.0.0.1:9200}"

EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-Qwen3-Embedding-8B}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-a}"

# Inference settings.
BCP_LINK_DATASET="${BCP_LINK_DATASET:-data/browsecomp_plus_decrypted_qa.jsonl}"
MODEL_TYPE="${MODEL_TYPE:-local}" # local or closed
MODEL_NAME="${MODEL_NAME:-your-model}"
LLM_BASE_URLS="${LLM_BASE_URLS:-[\"http://127.0.0.1:8000/v1\"]}"
LLM_API_KEY="${LLM_API_KEY:-a}" # Local servers commonly accept "a"; closed models require the provider key.

# Optional inference settings.
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-131072}"
RUN_DIR="${RUN_DIR:-outputs/bcp-link}"
GENERATION_DIR="${GENERATION_DIR:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-48}"
LLM_CONCURRENCY_LIMIT="${LLM_CONCURRENCY_LIMIT:-48}"
ES_MAX_CONCURRENCY="${ES_MAX_CONCURRENCY:-64}"
EMBEDDING_MAX_CONCURRENCY="${EMBEDDING_MAX_CONCURRENCY:-64}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"
CHECKPOINT_RESUME="${CHECKPOINT_RESUME:-true}"
CONTENT_FIELD="${CONTENT_FIELD:-text}"
MAX_ITEMS="${MAX_ITEMS:-}"

# Evaluation settings used by scripts/run_evaluate.sh.
JUDGE_MODEL="${JUDGE_MODEL:-qwen-32b}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8010/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-a}"
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-8}"
BENCHMARK_TOTAL="${BENCHMARK_TOTAL:-830}"
