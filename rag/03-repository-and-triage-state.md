# 03 — Repository and triage state

## Repository identity
- Canonical repository: `vitrixLab/bnn`.
- `main` branch existence was confirmed in this task.
- Current main README was fetched; blob SHA: `fdf5e952a42e7b67815db73dc780915351b32d65`.
- README describes an experimental C++17 research repo for low-bit quantization, packed matrix-vector kernels, and model/runtime engineering; prototype/not production-ready; quantized kernels not integrated into nanoLM inference. VERIFIED from README only.
- Corpus branch: `rag/conversation-corpus-2026-11-05`, created from `main` in this task.

## Accepted triage snapshot (REPORTED)
`TRIAGE.md v2.1`: A=2, B=0, C=10, D=16, E=10; total 38.
- PR #15/#16 reported verification passed, but required-check/branch-protection configuration not verified.
- #35/#31/#30/#20 closure candidates; later issue-body review reportedly found active/incomplete states.
- #29/#28/#27 protected from action under scoped directive.
- Category E decisions: #17, #13, #12, #8, #32, #26, #22.
- Documentation tasks: #37, #33, #36, #25, #24, #21.
- CLINE process defects: missing before-hash, unreliable editing narrative, inconsistent count.

## Research boundaries
- Issue #27 owns nanoLM baseline, dataset, training plan (per attached note).
- Do not extrapolate from ~14K-parameter nanoLM to SLM/MLM/LLM.
- INT4/INT8 rounding work (#21/#22/#24, PR #15, PR #23) and preservation-restricted PR #8 are outside the Algebraic Mind note.
- `binary-skill/` is treated in the note as a separate knowledge-base file format, not a neural-network component, unless owner says otherwise.

## Freshness
Triage and issue/PR states above are historical conversation claims, not a live recount in this corpus task. Live operations here were limited to main README fetch, branch search, and corpus branch creation.