# BNN conversation RAG corpus

**Repository:** `vitrixLab/bnn` · **Branch:** `rag/conversation-corpus-2026-11-05` · **Scope:** conversation and the three supplied attachments.

This is a retrieval-ready knowledge corpus, not a verbatim transcript, code certification, execution log, or external vector-store ingestion.

## Contents
- `00-scope-and-evidence-policy.md`
- `01-conversation-chronology.md`
- `02-decisions-and-authorizations.md`
- `03-repository-and-triage-state.md`
- `04-pr-verification-and-merge-gates.md`
- `05-issue-closure-analysis.md`
- `06-pr-solutions-import-history.md`
- `07-research-context-and-attached-sources.md`
- `08-followup-patch-source-analysis.md`
- `09-open-gates-and-next-actions.md`
- `evidence/source-manifest.md`
- `evidence/evidence-register.md`
- `metadata/entities.json`, `metadata/decisions.json`, `metadata/chunks.json`

## Retrieval rules
1. Preserve chronology; do not erase superseded claims.
2. Keep VERIFIED, REPORTED, UNKNOWN, PROPOSED, NOT AUTHORIZED, and NOT RUN distinct.
3. Agent reports remain REPORTED until independently checked.
4. Never infer authorization from tests, mathematical analysis, mergeability, or a document containing a close command.
5. The conversation context is reconstructed from available project context and is not a complete raw transcript export.

**Ingestion status: NO EXTERNAL RAG / VECTOR-STORE INGESTION HAS BEEN PERFORMED OR VERIFIED.** The branch contains corpus files only. Embedding/indexing is a separate future operation.