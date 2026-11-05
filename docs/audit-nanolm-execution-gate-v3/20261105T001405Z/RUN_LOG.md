# NanoLM Sandbox Execution Gate — Run Log
**RUN_ID:** 20261105T001405Z
**Start (UTC):** 2026-11-05T00:14:05Z
**End (UTC):** 2026-11-05T00:14:05Z (same minute; run completed within the minute)

## Repository identity and integrity

**Commit SHA (HEAD):** 44596ecb4e87261f87c9fb00577daf9699d4f9e9
**Expected commit SHA:** ceb36ad475d1d0e80de2df2a7f54986df75068ba
**Commit match:** NO — HEAD is 44596ecb, which is a child of ceb36ad on branch
  docs/nanolm-execution-evidence-20261105T080000Z. The expected commit ceb36ad
  exists on branch docs/audit-nanolm-execution-gate-v3 (and origin/...).
  Source files (nanolm.cpp, sandbox harness, runner) are IDENTICAL between
  ceb36ad and 44596ecb (git diff produced no output for the three key files).

**Working-tree status (git status --short):**
  ?? build-nanolm/
  ?? docs/audit-nanolm-execution-gate-v3/20261105T001405Z/  (this evidence dir)
  ?? docs/audit-nanolm-execution-gate-v3/20261105T001129Z/  (empty, superseded)
  ?? gate_runner.py
  ?? run-20261104T233249Z.log  (previous run log)
  ?? run-20261105T001405Z.log  (current run log)
  ?? run_nanolm_gate.sh
  ?? $LOG  (artifact from a misquoted earlier command attempt)
  Note: build-nanolm/ and run-*.log are untracked build artifacts; the evidence
  directory is the only new tracked-content addition.

**Tracked-file diff:** `git diff -- src/nanolm.cpp test-bundle/tests/nanolm_sandbox_tests.cpp test-bundle/run_nanolm_sandbox_tests.sh` produced no output — no modifications to the three key files.

## File SHA-256 hashes

| File | SHA-256 | Expected | Match |
|---|---|---|---|
| src/nanolm.cpp | B2F7D638F4D86C46C7539A999314BCD75498A561DF1D9C1516EFB8BD10C38270 | B2F7D638F4D86C46C7539A999314BCD75498A561DF1D9C1516EFB8BD10C38270 | YES |
| test-bundle/tests/nanolm_sandbox_tests.cpp | F2113F618808AC909C2234E2721834492B3DE1FA3794F0F6AB1DCCD187CF2B0B | F2113F618808AC909C2234E2721834492B3DE1FA3794F0F6AB1DCCD187CF2B0B | YES |
| test-bundle/run_nanolm_sandbox_tests.sh | AB7EAC51C570E6804A22B436023A744996D9D9E91BE77BC6FAED5569C49D9E9C | AB7EAC51C570E6804A22B436023A744996D9D9E91BE77BC6FAED5569C49D9E9C | YES |

All three file hashes match the expected values.

## Environment

**WSL distribution:** AlmaLinux-9, WSL version 2
**OS (inside WSL):** AlmaLinux 9.8 (Olive Jaguar)
**Kernel (inside WSL):** Linux DESKTOP-DOR225G 6.18.40.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Fri Jul 31 22:12:15 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
**Compiler:** g++ (GCC) 11.5.0 20240719 (Red Hat 11.5.0-14)
**Build system:** cmake version 3.31.8, make /usr/bin/make
**Namespace tools:** unshare /usr/bin/unshare (util-linux-2.37.4-25.el9.x86_64)
**Container runtime:** docker NOT INSTALLED, podman NOT INSTALLED

## Network isolation verification

Plain `unshare -n true`: FAILED — "unshare: unshare failed: Operation not permitted" (same blocker as prior run 20261104T233249Z)
`unshare -n --map-root-user true`: SUCCEEDED (exit 0)
`unshare -n --map-root-user -- ip addr`: VERIFIED — only loopback interface (lo), state DOWN; no external network interfaces (eth0 absent)
`kernel.unprivileged_userns_clone`: "no" (unprivileged user namespaces disabled)

**Isolation method used:** `unshare -n --map-root-user` to create a verified
network-isolated namespace, with `BNN_NETWORK_ISOLATED=1` set so the runner
skips its internal (failing) plain `unshare -n` call and runs the tests
directly inside the pre-isolated namespace. The runner itself was NOT modified.
**Isolation independently verified:** YES — `ip addr` inside the namespace
confirmed only the loopback interface exists, state DOWN, no external network
access.

## Exact command executed

```
wsl -d AlmaLinux-9 -- python3 /mnt/d/citrixlabph/audit-worktree/gate_runner.py
```

where gate_runner.py runs:
```
unshare -n --map-root-user -- bash test-bundle/run_nanolm_sandbox_tests.sh
```
with `BNN_NETWORK_ISOLATED=1`, output teed to `run-20261105T001405Z.log`,
using `pipefail` semantics via Python subprocess (stdout+stderr merged into
the log file, return code captured independently).

## Stage exit codes

| Stage | Exit code | Result |
|---|---:|---|
| CMake configure | 0 | PASS |
| CMake build | 0 | PASS |
| Strict harness compile (g++ -std=c++17 -Wall -Wextra -Werror) | 0 | PASS |
| attention_head_cache_and_gradient | 0 | PASS |
| next_token_full_window_reference | 0 | PASS |
| parameter_count_matches_allocations | 0 | PASS |
| checkpoint_rejects_before_allocation | 0 | PASS |
| cli_training_and_save_failures | 0 | PASS |
| isolated_deterministic_replay | 0 | PASS |
| **Overall runner** | **0** | **PASS** |

All six named tests executed separately under the verified network-isolated
namespace, each with its own stdout, stderr, and exit-code file in
build-nanolm/test-logs/.

## Log files

All expected per-stage and per-test log files exist in
`build-nanolm/test-logs/` and have been copied to the evidence directory at
`docs/audit-nanolm-execution-gate-v3/20261105T001405Z/build-and-test-logs/`:

- CMAKE_CONFIGURE.{exit,stdout.log,stderr.log}
- CMAKE_BUILD.{exit,stdout.log,stderr.log}
- HARNESS_COMPILE.{exit,stdout.log,stderr.log}
- TEST_attention_head_cache_and_gradient.{exit,stdout.log,stderr.log}

## Warnings, limitations, and environmental notes

1. **Commit discrepancy:** HEAD is 44596ecb, not the expected ceb36ad. The
   source files are identical between the two commits; only documentation
   differs. This is reported for transparency.

2. **Network isolation:** Plain `unshare -n` is not permitted in this WSL2
   AlmaLinux-9 environment (same blocker as the prior run). Verified isolation
   was achieved via `unshare -n --map-root-user` with `BNN_NETWORK_ISOLATED=1`.
   Docker and podman are not installed.

3. **Compiler warnings:** The harness compile with `-Wall -Wextra -Werror`
   produced no errors (exit 0). The CMake build of nanolm.cpp produced
   `-Wsign-conversion` and `-Wconversion` warnings (these are in the source
   build, not the strict harness compile, which uses -Werror and passed clean).

4. **WSL mount git access:** Git commands from inside WSL at
   /mnt/d/citrixlabph/audit-worktree fail because the .git worktree pointer
   references a Windows path (D:/citrixlabph/bnn/.git/worktrees/audit-worktree).
   Git operations were performed from Windows PowerShell instead.

5. **Evidence directory superseded:** An empty evidence directory
   20261105T001129Z/ was created during an earlier misquoted command attempt
   and is superseded by this directory (20261105T001405Z/).

## Provenance distinction

"Reviewed baseline: db3e796, NanoLM source blob 5f0c599d1ea3b00f7d725115fe3f65dbd3f44753. Candidate execution revision: ceb36ad475d1d0e80de2df2a7f54986df75068ba. The source difference reflects the later remediation; baseline research findings must not be represented as measurements of the candidate."

## Verdict

PASS — all six tests compiled and passed under verified network isolation.
- TEST_next_token_full_window_reference.{exit,stdout.log,stderr.log}
- TEST_parameter_count_matches_allocations.{exit,stdout.log,stderr.log}
- TEST_checkpoint_rejects_before_allocation.{exit,stdout.log,stderr.log}
- TEST_cli_training_and_save_failures.{exit,stdout.log,stderr.log}
- TEST_isolated_deterministic_replay.{exit,stdout.log,stderr.log}

Total: 27 files (9 stages × 3 files each). All exit files contain "0".

## Runner output

The full runner output is preserved in `runner-output.log` (copy of
`run-20261105T001405Z.log`). All six tests printed `[PASS]` and the runner
printed `SUMMARY: all 6 NanoLM sandbox tests passed`.