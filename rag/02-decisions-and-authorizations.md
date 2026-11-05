# 02 — Decisions and authorization ledger

## Explicit user requirements (VERIFIED, 2026-11-05)
- Destination: repository `vitrixLab/bnn`.
- Scope: conversation.
- Preserve provenance, chronology, decisions, authorizations, unresolved gates.
- Do not claim external ingestion.

## Governance constraints from prior BNN conversation (REPORTED)
- PRs generally remain OPEN/DRAFT/UNMERGED until an explicit gate is satisfied.
- Merge only after checking PR state, draft state, mergeability, applicable checks, branch protection/rulesets.
- No force-push, direct commit to protected branches, secrets, admin/force/bypass shortcuts.
- Issue closure is content-based: clear close recommendation, issue identified, rationale for closure now, follow-up location if applicable, no unresolved decision.
- A plan/spec is not completion evidence.
- Do not touch #29/#28/#27 or unrelated PRs/issues under the scoped directive.

## Authorization table
| Action | Status | Basis |
|---|---|---|
| Create corpus branch and add RAG files | AUTHORIZED | Explicit request to build corpus in repo. |
| Create/merge corpus PR | NOT AUTHORIZED / NOT PERFORMED | Destination selected, but PR not requested. |
| Merge #15/#16 | NOT AUTHORIZED | Protection verification remained UNKNOWN; no explicit merge approval. |
| Close #35/#31/#30/#20 | NOT AUTHORIZED / NOT PERFORMED | Issue bodies reportedly active/incomplete. |
| Modify #29/#28/#27, PR #8, T1 artifacts, code, CI, workflows, protection | NOT AUTHORIZED under earlier scoped directive | Preserve prior boundaries. |
| Run experiments/training/benchmarks or publish probes/checkpoints | NOT AUTHORIZED by research note | BG gates and #27 review apply. |
| External vector DB ingestion | NOT PERFORMED | No external target/indexer action occurred; user says do not claim it. |

## Supersession
Retain earlier proposals as history when displaced by later instructions or stronger evidence. Do not erase chronology.