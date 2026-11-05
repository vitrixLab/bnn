# 09 — Open gates and next actions

## Repository / governance
- **G1 — Corpus PR:** branch created. PR not created; user selected repo destination but did not explicitly request PR creation.
- **G2 — pr-solutions remote verification:** UNKNOWN; local CLINE report conflicts with later reported 422/404. Verify local/remote refs before relying on it.
- **G3 — PR #15/#16:** HOLD; protection/required-check verification remained UNKNOWN due reported 403. Recheck only with suitable permissions and explicit owner authorization.
- **G4 — #35/#31/#30/#20:** leave open pending current direct issue/rationale review and valid closure evidence.
- **G5 — Protected scope:** do not touch #29/#28/#27, PR #8, or T1 artifacts under earlier scoped directive.

## Research / technical
- **G6 — Patch application:** no evidence patch `5059af8…` is applied to current main or a branch; verify reachability and exact diff before application.
- **G7 — Patch validation:** build/tests/CI/benchmark oracle/provenance/negative tests NOT RUN in this corpus task.
- **G8 — nanoLM baseline:** #27 prerequisites (pinned vocabulary, held-out BPC eval, run logging) must precede BG0; no training/probes authorized by note.
- **G9 — BIN1 parity:** inspect kernel numerics/layout/scale/packing and establish parity before experiments; UNKNOWN in note.
- **G10 — System 3-A:** verify source rows and close O2–O5; confirm power in G*Power; resolve remaining high-priority items before pilot/pre-registration. PDF says not pilot-ready.

## RAG operational status
These files are retrieval-ready source material, not embeddings or a live retrieval service. No external ingestion has been performed or verified. Future indexer should retain source_id, source SHA, evidence status, event date, and authorization scope in every chunk.