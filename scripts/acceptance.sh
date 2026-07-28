#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_UV="0.6.14"
EXPECTED_PYTHON="3.11.12"
EXPECTED_SCHEMA="9"
EXPECTED_SCHEMA_FINGERPRINT="34a66995fe65ac46616e49767bbe9f8c15a95f764397ab31240c907f5624ff10"
MODE="full"

if [[ "${1:-}" == "--quick" ]]; then
  MODE="quick"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--quick]" >&2
  exit 2
fi

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

for command in curl git uname uv; do
  command -v "$command" >/dev/null 2>&1 || fail "required command is unavailable: $command"
done

HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
case "$HOST_OS" in
  Darwin)
    command -v sw_vers >/dev/null 2>&1 || fail "sw_vers is unavailable on macOS"
    HOST_VERSION="$(sw_vers -productVersion)"
    HOST_MAJOR="${HOST_VERSION%%.*}"
    (( HOST_MAJOR >= 13 )) || fail "core dependencies require macOS 13 or newer"
    ;;
  Linux)
    HOST_VERSION="$(uname -r)"
    ;;
  *)
    fail "the POSIX acceptance script supports Linux and macOS, got $HOST_OS"
    ;;
esac

UV_OUTPUT="$(uv --version)"
UV_ACTUAL="${UV_OUTPUT#uv }"
UV_ACTUAL="${UV_ACTUAL%% *}"
[[ "$UV_ACTUAL" == "$EXPECTED_UV" ]] || fail "expected uv $EXPECTED_UV, got $UV_OUTPUT"

ACCEPT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dithyramba-acceptance.XXXXXX")"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  if [[ "${DITHYRAMBA_KEEP_ACCEPTANCE:-0}" == "1" ]]; then
    echo "Acceptance evidence retained at: $ACCEPT_ROOT"
  else
    rm -rf "$ACCEPT_ROOT"
  fi
}
trap cleanup EXIT

umask 077
mkdir -p \
  "$ACCEPT_ROOT/dist" \
  "$ACCEPT_ROOT/python" \
  "$ACCEPT_ROOT/tmp" \
  "$ACCEPT_ROOT/uv-cache"
export TMPDIR="$ACCEPT_ROOT/tmp"
export UV_CACHE_DIR="$ACCEPT_ROOT/uv-cache"
export UV_LINK_MODE="copy"
export UV_MANAGED_PYTHON="1"
export UV_PYTHON_INSTALL_DIR="$ACCEPT_ROOT/python"

cd "$ROOT"
GIT_TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || fail "acceptance must run from a Git checkout"
GIT_TOPLEVEL="$(cd "$GIT_TOPLEVEL" && pwd -P)"
[[ "$GIT_TOPLEVEL" == "$ROOT" ]] \
  || fail "checkout root mismatch: expected $ROOT, got $GIT_TOPLEVEL"
COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null)" || fail "HEAD does not resolve to a commit"
GIT_STATUS="$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)" \
  || fail "could not inspect Git checkout status"
if [[ -n "$GIT_STATUS" ]]; then
  echo "FAIL: Git checkout is not clean:" >&2
  printf '%s\n' "$GIT_STATUS" >&2
  exit 1
fi
printf 'commit_sha=%s\n' "$COMMIT_SHA" | tee "$ACCEPT_ROOT/acceptance-metadata.log" >/dev/null

verify_checkout_unchanged() {
  local current_sha current_status
  current_sha="$(git rev-parse HEAD 2>/dev/null)" || fail "HEAD no longer resolves to a commit"
  [[ "$current_sha" == "$COMMIT_SHA" ]] \
    || fail "HEAD changed during acceptance: expected $COMMIT_SHA, got $current_sha"
  current_status="$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)" \
    || fail "could not re-inspect Git checkout status"
  if [[ -n "$current_status" ]]; then
    echo "FAIL: acceptance changed the Git checkout:" >&2
    printf '%s\n' "$current_status" >&2
    exit 1
  fi
}

echo "Dithyramba clean-install acceptance"
echo "  commit:   $COMMIT_SHA"
echo "  host:     $HOST_OS $HOST_VERSION"
echo "  arch:     $HOST_ARCH"
echo "  uv:       $UV_OUTPUT"
echo "  Python:   $EXPECTED_PYTHON"
echo "  mode:     $MODE"
echo "  scratch:  $ACCEPT_ROOT"

uv python install "$EXPECTED_PYTHON"
PYTHON_BIN="$(uv python find "$EXPECTED_PYTHON")"
PYTHON_ACTUAL="$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
PYTHON_ARCH="$($PYTHON_BIN -c 'import platform; print(platform.machine())')"
[[ "$PYTHON_ACTUAL" == "$EXPECTED_PYTHON" ]] || fail "managed Python is $PYTHON_ACTUAL"
[[ "$PYTHON_ARCH" == "$HOST_ARCH" ]] \
  || fail "managed Python architecture $PYTHON_ARCH differs from host $HOST_ARCH"
export UV_PYTHON_DOWNLOADS="never"

"$PYTHON_BIN" scripts/release_audit.py
uv lock --check

VERSION="$($PYTHON_BIN -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
SDIST="$ACCEPT_ROOT/dist/dithyramba-$VERSION.tar.gz"
WHEEL="$ACCEPT_ROOT/dist/dithyramba-$VERSION-py3-none-any.whl"

# Build the wheel from the just-built sdist, never directly from the checkout.
uv build --sdist --no-sources --python "$PYTHON_BIN" --out-dir "$ACCEPT_ROOT/dist" "$ROOT"
[[ -f "$SDIST" ]] || fail "expected sdist was not produced: $SDIST"
uv build --wheel --no-sources --python "$PYTHON_BIN" --out-dir "$ACCEPT_ROOT/dist" "$SDIST"
[[ -f "$WHEEL" ]] || fail "expected wheel was not produced: $WHEEL"

"$PYTHON_BIN" - "$SDIST" "$WHEEL" <<'PY' | tee "$ACCEPT_ROOT/SHA256SUMS"
import hashlib
import sys

for value in sys.argv[1:]:
    digest = hashlib.sha256()
    with open(value, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"{digest.hexdigest()}  {value}")
PY
"$PYTHON_BIN" scripts/inspect_distribution.py --source-root "$ROOT" "$WHEEL" "$SDIST"

# Create a base-only environment, exclude the editable checkout, prohibit third-party builds,
# and install only the wheel under acceptance.
RUNTIME_ENV="$ACCEPT_ROOT/runtime-venv"
UV_PROJECT_ENVIRONMENT="$RUNTIME_ENV" \
  uv sync --frozen --no-dev --no-install-project --no-build --python "$PYTHON_BIN"
uv pip install --python "$RUNTIME_ENV/bin/python" --no-deps --no-build "$WHEEL"
uv pip check --python "$RUNTIME_ENV/bin/python"

RUNTIME_PYTHON="$RUNTIME_ENV/bin/python"
RUNTIME_CLI="$RUNTIME_ENV/bin/dithyramba"
[[ "$($RUNTIME_CLI --version)" == "$VERSION" ]] || fail "installed CLI version mismatch"

env -u PYTHONPATH DITHYRAMBA_SOURCE_ROOT="$ROOT" "$RUNTIME_PYTHON" - <<'PY'
import os
import sqlite3
import sys
from pathlib import Path

import dithyramba

package = Path(dithyramba.__file__).resolve()
source = Path(os.environ["DITHYRAMBA_SOURCE_ROOT"]).resolve()
assert Path(sys.prefix).resolve() in package.parents, package
assert source not in package.parents, package
connection = sqlite3.connect(":memory:")
connection.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(value)")
connection.close()
PY

env -u PYTHONPATH "$RUNTIME_PYTHON" -m dithyramba.store.migrations verify \
  | tee "$ACCEPT_ROOT/migrations.json"
env -u PYTHONPATH \
  EXPECTED_SCHEMA="$EXPECTED_SCHEMA" \
  EXPECTED_SCHEMA_FINGERPRINT="$EXPECTED_SCHEMA_FINGERPRINT" \
  "$RUNTIME_PYTHON" - "$ACCEPT_ROOT/migrations.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
assert payload["status"] == "ok"
assert payload["schema_version"] == int(os.environ["EXPECTED_SCHEMA"])
assert payload["schema_fingerprint"] == os.environ["EXPECTED_SCHEMA_FINGERPRINT"]
PY

DEMO_DATA="$ACCEPT_ROOT/demo-data"
env -u PYTHONPATH "$RUNTIME_PYTHON" "$ROOT/scripts/demo.py" --data-home "$DEMO_DATA" \
  | tee "$ACCEPT_ROOT/demo.log"
LIBRARY_ID="$(sed -n 's/^  library_id:[[:space:]]*//p' "$ACCEPT_ROOT/demo.log" | tail -1)"
RESTORED_DATA="$(sed -n 's/^  restored_data_home:[[:space:]]*//p' "$ACCEPT_ROOT/demo.log" | tail -1)"
[[ -n "$LIBRARY_ID" ]] || fail "demo did not report a Library ID"
[[ -n "$RESTORED_DATA" ]] || fail "demo did not report its restored data root"
grep -Eq 'Sources indexed \([1-9][0-9]* processed\)' "$ACCEPT_ROOT/demo.log" \
  || fail "demo did not process a source"
grep -Eq 'Evidence packet persisted \([1-9][0-9]* fragment\(s\)\)' "$ACCEPT_ROOT/demo.log" \
  || fail "demo did not persist source evidence"

# Replay the public 1,000-fragment evidence path from the installed wheel.
env -u PYTHONPATH "$RUNTIME_PYTHON" -m verification.run_public_replay \
  --warmups 1 --measured 2 --output "$ACCEPT_ROOT/public-replay.json"
env -u PYTHONPATH "$RUNTIME_PYTHON" - "$ACCEPT_ROOT/public-replay.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
assert payload["schema"] == "dithyramba.public_replay_result/1.0"
assert payload["verification_id"] == "public_replay_1000_v1"
assert payload["workload"]["fragment_count"] == 1_000
assert payload["result"]["candidate_count"] == 1
assert payload["result"]["packet_hash_count"] == 1
assert payload["result"]["provider_call_count"] == 0
PY

for DATA_HOME in "$DEMO_DATA" "$RESTORED_DATA"; do
  env -u PYTHONPATH "$RUNTIME_CLI" doctor \
    --library "$LIBRARY_ID" --data-home "$DATA_HOME" --json \
    >"$ACCEPT_ROOT/doctor-$(basename "$DATA_HOME").json"
  env -u PYTHONPATH \
    EXPECTED_SCHEMA="$EXPECTED_SCHEMA" \
    EXPECTED_SCHEMA_FINGERPRINT="$EXPECTED_SCHEMA_FINGERPRINT" \
    "$RUNTIME_PYTHON" - "$ACCEPT_ROOT/doctor-$(basename "$DATA_HOME").json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
assert payload["status"] == "ok"
assert payload["schema_version"] == int(os.environ["EXPECTED_SCHEMA"])
assert payload["schema_fingerprint"] == os.environ["EXPECTED_SCHEMA_FINGERPRINT"]
assert payload["fts5_available"] is True
assert payload["external_services_contacted"] is False
PY
done

# Exercise the real loopback server from the installed wheel.
PORT="$($RUNTIME_PYTHON - <<'PY'
import socket

with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    print(probe.getsockname()[1])
PY
)"
env -u PYTHONPATH "$RUNTIME_CLI" serve \
  --library "$LIBRARY_ID" --data-home "$DEMO_DATA" --port "$PORT" \
  >"$ACCEPT_ROOT/serve.stdout" 2>"$ACCEPT_ROOT/serve.stderr" &
SERVER_PID="$!"
READY=0
for ((attempt = 0; attempt < 100; attempt += 1)); do
  if curl --fail --silent --show-error --max-time 2 \
    "http://127.0.0.1:$PORT/health" >"$ACCEPT_ROOT/health.json" 2>/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
[[ "$READY" == "1" ]] || fail "loopback server did not become healthy"
env -u PYTHONPATH \
  EXPECTED_LIBRARY_ID="$LIBRARY_ID" EXPECTED_SCHEMA="$EXPECTED_SCHEMA" \
  "$RUNTIME_PYTHON" - "$ACCEPT_ROOT/health.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
assert payload["library_id"] == os.environ["EXPECTED_LIBRARY_ID"]
assert payload["schema_version"] == int(os.environ["EXPECTED_SCHEMA"])
PY
kill "$SERVER_PID"
wait "$SERVER_PID" >/dev/null 2>&1 || true
SERVER_PID=""

if [[ "$MODE" == "quick" ]]; then
  verify_checkout_unchanged
  echo "PASS: quick artifact/runtime route completed"
  echo "NOTE: Chromium, PDF subprocess, and full aggregate remain for full mode"
  exit 0
fi

# Full mode uses a second clean environment containing locked development dependencies.
QA_ENV="$ACCEPT_ROOT/qa-venv"
UV_PROJECT_ENVIRONMENT="$QA_ENV" \
  uv sync --frozen --dev --no-install-project --no-build --python "$PYTHON_BIN"
uv pip install --python "$QA_ENV/bin/python" --no-deps --no-build "$WHEEL"
uv pip check --python "$QA_ENV/bin/python"

QA_PATH="$QA_ENV/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$ACCEPT_ROOT/playwright-browsers"
env -u PYTHONPATH PATH="$QA_PATH" "$QA_ENV/bin/playwright" install chromium

cd "$ROOT"
env -u PYTHONPATH PATH="$QA_PATH" "$QA_ENV/bin/python" -m pytest \
  tests/unit/test_p2_parsers.py::test_pdf_is_parsed_out_of_process_with_word_union_bbox \
  --no-cov -q -rA -p no:cacheprovider | tee "$ACCEPT_ROOT/pdf-test.log"
if grep -Eiq '(^|[[:space:]])[0-9]+ skipped' "$ACCEPT_ROOT/pdf-test.log"; then
  fail "the required PDF subprocess test was skipped"
fi

env -u PYTHONPATH PATH="$QA_PATH" "$QA_ENV/bin/python" -m pytest \
  tests/browser --no-cov -q -rA -p no:cacheprovider | tee "$ACCEPT_ROOT/browser-tests.log"
if grep -Eiq '(^|[[:space:]])[0-9]+ skipped' "$ACCEPT_ROOT/browser-tests.log"; then
  fail "browser acceptance contained a skip"
fi

env -u PYTHONPATH PATH="$QA_PATH" "$QA_ENV/bin/check"

verify_checkout_unchanged
echo "PASS: full clean-install source/sdist/wheel/runtime/browser/PDF acceptance completed"
