#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-nanolm}"
LOG_DIR="${LOG_DIR:-$BUILD_DIR/test-logs}"
BIN="$BUILD_DIR/run_tests"
mkdir -p "$BUILD_DIR" "$LOG_DIR"
run_logged() {
  local label="$1"; shift
  local stdout_file="$LOG_DIR/${label}.stdout.log"
  local stderr_file="$LOG_DIR/${label}.stderr.log"
  local exit_file="$LOG_DIR/${label}.exit"
  set +e; "$@" >"$stdout_file" 2>"$stderr_file"; local rc=$?; set -e
  cat "$stdout_file"; cat "$stderr_file" >&2
  printf 'EXIT_%s=%s\n' "$label" "$rc"; printf '%s\n' "$rc" >"$exit_file"
  return "$rc"
}
run_logged CMAKE_CONFIGURE cmake -S "$ROOT/src" -B "$BUILD_DIR"
run_logged CMAKE_BUILD cmake --build "$BUILD_DIR"
run_logged HARNESS_COMPILE g++ -std=c++17 -Wall -Wextra -Werror -I "$ROOT/src" \
  "$ROOT/test-bundle/tests/nanolm_sandbox_tests.cpp" -o "$BIN"
tests=(attention_head_cache_and_gradient next_token_full_window_reference parameter_count_matches_allocations checkpoint_rejects_before_allocation cli_training_and_save_failures isolated_deterministic_replay)
for test_name in "${tests[@]}"; do
  if [[ "${BNN_NETWORK_ISOLATED:-0}" == "1" ]]; then
    run_logged "TEST_$test_name" "$BIN" --test "$test_name"
  else
    if ! command -v unshare >/dev/null 2>&1; then
      echo "ERROR: unshare is required unless BNN_NETWORK_ISOLATED=1 is set by an isolated container runtime." >&2
      exit 2
    fi
    run_logged "TEST_$test_name" unshare -n -- "$BIN" --test "$test_name"
  fi
done
echo "SUMMARY: all 6 NanoLM sandbox tests passed"
