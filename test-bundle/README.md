# NanoLM sandbox test bundle

This bundle builds NanoLM with CMake, compiles a C++17 regression harness with
GCC, and executes six focused tests. The harness includes `src/nanolm.cpp` in
one translation unit to exercise internal functions without widening visibility.

## Requirements

Linux x86-64 (native Linux or WSL2), GCC/G++ with C++17, CMake 3.14+, Bash, and
either `unshare -n` permission or Docker with `--network none`. The runner fails
closed if no isolation boundary is supplied.

## WSL2 / native Linux

From the repository root:

```bash
git checkout main
cmake --version
g++ --version
unshare -n -- true
./test-bundle/run_nanolm_sandbox_tests.sh
```

The runner executes:

```bash
cmake -S src -B build-nanolm
cmake --build build-nanolm
g++ -std=c++17 -Wall -Wextra -Werror -I src \
  test-bundle/tests/nanolm_sandbox_tests.cpp -o build-nanolm/run_tests
```

Each test runs separately under `unshare -n`; stdout, stderr, and exit codes are
stored in `build-nanolm/test-logs/`. If WSL cannot create network namespaces,
use Docker rather than running unisolated.

## Docker network isolation

Build the image while registry/package access is available:

```bash
docker build -f test-bundle/Dockerfile -t bnn-nanolm-sandbox .
```

Run with networking disabled:

```bash
docker run --rm --network none \
  -e BNN_NETWORK_ISOLATED=1 \
  -v "$PWD:/workspace" -w /workspace \
  bnn-nanolm-sandbox
```

The image contains GCC and CMake. Docker's `--network none` is the isolation
boundary; `BNN_NETWORK_ISOLATED=1` is accepted only for this explicit container
invocation and avoids nested `unshare`, which is commonly prohibited in Docker.

## Test inventory

1. `attention_head_cache_and_gradient` — finite-difference Q/K/V gradient checks
   across multiple attention heads.
2. `next_token_full_window_reference` — verifies all T next-token targets and
   a gradient on the final positional embedding row.
3. `parameter_count_matches_allocations` — checks accounting with a non-default
   layer count.
4. `checkpoint_rejects_before_allocation` — rejects a truncated checkpoint
   without mutating the destination model.
5. `cli_training_and_save_failures` — verifies save failure propagates to a
   non-zero CLI exit.
6. `isolated_deterministic_replay` — verifies save/load greedy generation is
   repeatable.

To run one test on WSL/Linux after compilation:

```bash
unshare -n -- build-nanolm/run_tests --test attention_head_cache_and_gradient
```

Do not claim the execution gate is satisfied unless CMake, strict GCC compilation,
all six tests, and network isolation pass. An unavailable isolation command is
NOT RUN, not a test pass.
