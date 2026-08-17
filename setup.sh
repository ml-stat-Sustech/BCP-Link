#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v bash >/dev/null 2>&1 || fail "Bash is required."
command -v curl >/dev/null 2>&1 || fail "curl is required."
command -v python3 >/dev/null 2>&1 || fail "Python 3.12 or newer is required."

python3 - <<'PY' || fail "Python 3.12 or newer is required."
import sys

if sys.version_info < (3, 12):
    raise SystemExit(1)
PY

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found; installing it with the official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${UV_INSTALL_DIR:-${HOME}/.local/bin}:${HOME}/.cargo/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || fail "uv installation completed but uv is not on PATH."

echo "Synchronizing the Python environment."
uv sync

SETTINGS_FILE="scripts/settings.sh"
if [[ ! -e "${SETTINGS_FILE}" ]]; then
  cp scripts/settings.example.sh "${SETTINGS_FILE}"
  chmod 600 "${SETTINGS_FILE}"
  echo "Created ${SETTINGS_FILE} from the release template."
else
  echo "Keeping existing ${SETTINGS_FILE}."
fi

echo "Checking shell scripts and SearcherKit configuration."
bash -n setup.sh scripts/run_index.sh scripts/run_inference.sh \
  scripts/run_evaluate.sh scripts/settings.example.sh "${SETTINGS_FILE}"
uv run python -m searcherkit --help >/dev/null
uv run python -m searcherkit plugins deploy browsecomp-plus-link --help >/dev/null
uv run python -m searcherkit inspect --config-path searcherkit >/dev/null

placeholder_lines="$(
  grep -nE '(/path/to/|your-[a-z-]+|replace-with-your-api-key)' "${SETTINGS_FILE}" || true
)"
if [[ -n "${placeholder_lines}" ]]; then
  echo
  echo "Setup completed. Replace these placeholders before running an evaluation:"
  printf '%s\n' "${placeholder_lines}"
else
  echo "Setup completed; no release-template placeholders remain in ${SETTINGS_FILE}."
fi

echo "External Elasticsearch, embedding, model, and judge services were not contacted or started."
