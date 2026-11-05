# 00 — Scope and evidence policy

## Scope
This corpus captures the BNN-focused conversation context available for this task and the three attachments supplied in this conversation: `bnn-followup.patch`, `algebraic-mind-binary-finetuning.md`, and the System 3-A scoping-review/design-specification PDF. It excludes unrelated projects and general profile memories.

## Completeness limitation
Available context includes summarized prior turns and explicit omissions. It is not a complete raw transcript export. Chronology and earlier project facts are reconstructed from visible conversation summaries/project context. Do not label this a verbatim or exhaustive transcript.

## Status vocabulary
- **VERIFIED** — directly inspected in a supplied artifact or returned by a live repository tool in this task.
- **REPORTED** — stated by an agent, earlier audit, or prior conversation summary but not independently rechecked here.
- **UNKNOWN** — missing, inaccessible, contradictory, or insufficient evidence.
- **PROPOSED** — plan/design, not execution.
- **NOT AUTHORIZED** — outside explicit permission.
- **NOT RUN** — no execution evidence.
- **SUPERSEDED** — displaced by later explicit instruction or stronger evidence.

## Evidence hierarchy
1. Current direct user instruction and explicit authorization.
2. Directly inspected artifact or live repository response with locator/date.
3. Earlier project report with provenance, marked REPORTED.
4. Model inference, explicitly marked INFERENCE.

## Governance invariants
- Chat is not the source of truth for repository state.
- Evidence → mathematical analysis → derived evidence ≠ authorization.
- Passing checks do not prove branch-protection/ruleset requirements are satisfied.
- Classic branch-protection 404 does not prove absence of rulesets; 403/auth failure leaves protection UNKNOWN.
- No merge, issue closure, bypass, force-push, or protected-branch direct commit without explicit authorization and verified gates.
- BNN is an experimental research repository, not a production-ready quantized runtime.