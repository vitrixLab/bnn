# 08 — Follow-up patch source analysis

## Source metadata (VERIFIED from attachment)
Git-format patch header:
- commit header: `5059af8242d32246a1231659c929023eff99f26a`
- author: `claude <noreply@anthropic.com>`
- date: Wed, 28 Oct 2026 07:18:04 +0000
- subject: `fix(benchmark): build-system provenance, oracle-checked results, hardened CI validation`

This metadata does not prove commit reachability or patch application.

## Eight changed paths (from patch diff)
1. `.github/workflows/cpp-ci.yml` — benchmark verification, tracked JSON validation, CLI negative tests/artifacts; `BNN_BENCH_CONTEXT=ci-smoke`.
2. `.gitignore` — three-line change; inspect full diff before describing exact behavior.
3. `binary-skill/CMakeLists.txt` — build/source provenance context.
4. `binary-skill/benchmark.cpp` — remove hardcoded commit claim; oracle/fixture checks; batch sub-resolution operations; nearest-rank p95; JSON-write error handling.
5. `binary-skill/benchmark_results.json` — historical record marked `commit=null`, `historical_unverified`.
6. `binary-skill/tools/cli_negative_tests.py` — new tamper/truncation/bad-magic/wrong-hash tests.
7. `binary-skill/tools/validate_benchmark.py` — tracked/CI benchmark validator.
8. `evidence.md` — revise unsupported independence/controlled claims; document provenance/security boundary.

Patch summary: 8 files changed, 812 insertions, 317 deletions; two Python files created. These are patch statistics, not proof of application or CI success.

## Verification limits
Patch header and diff structure were inspected locally. Patch was not applied; tests/build/benchmarks were not run; GitHub reachability of `5059af8…` was not checked. Treat behavior as intended patch change, not verified runtime behavior.