# Evidence register

| ID | Claim | Status | Provenance | Limitation |
|---|---|---|---|---|
| EV-001 | `main` branch exists. | VERIFIED | GH-OBS-001 | Branch search did not return SHA. |
| EV-002 | README says experimental C++17 low-bit quantization repo, prototype/not production-ready, quantized kernels not integrated into nanoLM inference. | VERIFIED | README fetched from main; blob `fdf5e952a42e7b67815db73dc780915351b32d65` | README only; no broader audit. |
| EV-003 | Corpus branch created from main. | VERIFIED | GitHub branch-create response, 2026-11-05 | No PR/merge implied. |
| EV-004 | PR #15/#16 OPEN/non-draft/MERGEABLE/16 of 16 checks passed. | REPORTED | CLINE prior report | Not rechecked here. |
| EV-005 | Classic protection endpoint returned 403; required-check configuration UNKNOWN. | REPORTED | CLINE prior report | Permission limitation, not absence of protection. |
| EV-006 | Rulesets endpoint returned empty list. | REPORTED | CLINE prior report | Not independently rechecked. |
| EV-007 | #35/#31/#30/#20 bodies indicated planning/wait/open/active. | REPORTED | CLINE issue-body inspection | Not fetched in this task. |
| EV-008 | `docs/import-pr-solutions` commit `17dbd5b…` reported locally. | REPORTED | CLINE report | Remote later reportedly returned 422/404; status UNKNOWN. |
| EV-009 | Patch metadata and eight changed paths; 812 insertions/317 deletions. | VERIFIED as attachment content | SRC-001 | Not applied/tested. |
| EV-010 | Algebraic Mind note PLANNING / NOT VERIFIED; no experiment executed for note. | VERIFIED as document content | SRC-002 | Not independent repo audit. |
| EV-011 | System 3-A report says all counted rows U and design not pilot-ready; O2–O5 and G*Power open. | VERIFIED as document content | SRC-003 | Document's own claims, not independent literature verification. |
| EV-012 | Full chat externally ingested into RAG/vector DB. | NOT CLAIMED / NOT PERFORMED | No evidence | Explicitly do not claim. |

## Conflicts
- CLINE local import report conflicts with later remote 422/404 report; remote status UNKNOWN.
- Triage counts/issue states are historical, not current live state.
- Patch commit `5059af8…` is distinct from import commit `17dbd5b…`; do not conflate.
- Patch From SHA is not proof of reachability/base in current GitHub history.