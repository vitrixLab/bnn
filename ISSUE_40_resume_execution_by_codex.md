# Issue #40: Resume Execution by Codex

**Title:** Resume execution by codex  
**Issue:** #40  
**Status:** New  
**Category:** CORE_ENGINEERING  

## Summary
This issue tracks the implementation of resuming execution workflows using codex (OpenAI's code execution model). The goal is to enable seamless continuation of interrupted or paused computational workflows, particularly for nanolm training and skill distillation processes.

## Context
- Related to skill #44 `resume-codex` in the binary skill database
- Related to skill #43 `resume-claude` in the binary skill database
- Part of the broader evidence & benchmark verification framework

## Acceptance Criteria
- [ ] Implement resume functionality for nanolm training pipelines
- [ ] Integrate codex API for state restoration
- [ ] Ensure binary skill `#44` (`resume-codex`) is properly invoked
- [ ] Maintain evidence integrity during resume operations
- [ ] Update issue #40 tracking with progress

## Technical Notes
- Resume execution should leverage the binary skill format (`VSKB0001`)
- Must compatible with existing checkpoint versioning system
- Should work within the sandbox execution gate v3 constraints
- Needs to handle both GPU and CPU resume scenarios

## Linked Issues
- Issue #43: `resume-claude` - Claude-based resume implementation
- Issue #44: `resume-codex` - Codex-based resume implementation
- Planning doc: `HYPERION_TO_GPT1_PLAN.md` - Issue #40 tracking reference

## Comments
*(Retracted - actual GitHub issue #40 comment from codex takes precedence)*

### Claude Cross-Review Findings (Issue #40 comment 6828090894, 2026-11-12T15:36:58Z)
*Incorporated from GitHub issue #30 architecture discrepancy register; supplements comment #6827996122; does not supersede it.*

**Relevant independent cross-review points added to Issue #40:**

- **Parameter count:** ~810,784 claimed versus ~1,228,800 in the implementation discussion; resolve from actual allocated trainable tensors and add a count assertion.

- **Context:** 128 in the model description/config versus 64 in a CLI example; trace defaults and train/generation paths, then test consistency.

- **Data-size rule:** `params × 60` gives ~48.6 MB decimal for ~810K parameters, not ~5 MB. The heuristic itself remains unvalidated and must not be described as a minimum-data law.

- **GPT-1 likeness:** resume/checkpoint work is tooling and reproducibility infrastructure, not a direct likeness axis; full-state resume is an inferred prerequisite for long, reproducible recipe experiments.

- **Replay:** until full-state resume exists, deterministic replay means a fresh uninterrupted seeded run, not a weights-only restart.

- **Loss trace:** the existing Issue #40 comments mention corpus-size/generation observations but do not provide a raw loss trace or define whether loss is per byte or per token. OpenCode should link the already-existing trace and its logging definition/run metadata; do not invent the unit or discard local results.

*Claude independently confirmed the `0f889c0` specification/roadmap description and unchecked acceptance boxes, but did not read `6568e33` or the source. Its source-level non-verification is retained as a limitation of that review, not a contradiction of the source findings already recorded against the cited commit.*

### OpenCode Directive Findings (Issue #40 comment 6828568649, 2026-11-12T15:55:48Z)
*Repository-wide Markdown architecture reconciliation and implementation upgrade directive. Expands the work from three edited documents to a repository-wide, evidence-led documentation and architecture reconciliation. It does not authorize speculative claims or a blanket rewrite.*

**Governance:** Apply mandatory canonical rules from [Issue #25](https://github.com/vitrixLab/bnn/issues/25) and its linked Issue #20 ledger. Preserve scope, provenance, authorship, evidence labels, existing useful content, and local inference/training artifacts. Do not invent missing policy, test results, measurements, loss traces, skill IDs, or implementation status. Use exact full commit SHAs for source claims. Distinguish VERIFIED / OBSERVED / REPORTED / INFERRED / UNKNOWN / NOT RUN / NOT AUTHORIZED. A code path observed in source is not runtime proof.

**Trigger / baseline:** commit [0b7bf971bbaf43aad7e4e00af28b1e6925857212](https://github.com/vitrixLab/bnn/commit/0b7bf971bbaf43aad7e4e00af28b1e6925857212). This directive operationalizes its six cross-review findings. It expands the work from three edited documents to a repository-wide, evidence-led documentation and architecture reconciliation. It does **not** authorize speculative claims or a blanket rewrite.

**Mission:** Turn the findings recorded by commit 0b7bf97 into a coherent, technically accurate system model across all relevant repository Markdown documentation, then translate documentation-discovered architectural gaps into bounded implementation and regression-test work. The desired result is one consistent source of truth for nanoLM/Hyperion architecture, training, data, checkpoints/resume, reproducibility, evaluation, and GPT-1 comparison—not duplicated, contradictory mini-specificifications.

### Phase 0 — establish the actual repository baseline

1. Confirm current branch, HEAD SHA, working tree state, and relationship of HEAD to 0b7bf97. Do not assume the commit is current HEAD or on the active feature branch.
2. Inventory every tracked `.md` / `.mdx` file (including root, `docs/`, research, architecture, skills, issue indexes, and templates). Record path, current SHA, and whether it is authoritative, derivative, historical, or archival.
3. Identify canonical governance docs and existing source-of-truth documents before editing. Issue #25 is the rules index; Issue #20 is the coordination/evidence ledger. Respect any more specific canonical rules.
4. Search for every occurrence of the six findings and related stale claims: parameter counts/formulas, context/block size, vocab/tokenizer, data-size formulas, GPT-1 equivalence, deterministic replay, loss units/traces, checkpoint/resume, optimizer/scheduler/RNG state, benchmark/time/size claims.
5. Produce a compact impact matrix: document → claim(s) → evidence source/SHA → required action → dependency. Do not edit until this inventory exists.

### Phase 1 — establish implementation truth before changing prose

Inspect the actual code at the verified HEAD and relevant history. Trace, with file/line references:

- Config defaults and validation; CLI argument parsing and displayed examples.
- Tensor allocations and trainable parameter ownership, including attention/FFN projections, biases, LayerNorm, token/position embeddings, output head, and tied/untied weights.
- Training sequence construction, byte/token accounting, truncation, context-window enforcement, and generation path.
- Optimizer implementation/state, LR schedule/warmup/step counter, gradient/scaler state if applicable.
- Checkpoint format/version/header, serialization/deserialization offsets, compatibility checks, and corruption handling.
- RNG engines/seeds/states, sampler/data cursor, shuffle/order, batching, dataset identity/hash, and iteration counters.
- Existing loss logging, trace files, evaluation scripts, and actual local training/inference artifacts.

For every conclusion, cite the exact source SHA and path/line. If runtime behavior is needed, run the relevant test/experiment and retain command, environment, exit code, logs, and artifact hashes. If unavailable, label it NOT RUN or UNKNOWN—do not convert source inspection into a runtime claim.

### Phase 2 — resolve the six findings as testable contracts

**A. Parameter count**
- Resolve ~810,784 versus ~1,228,800 from actual allocated trainable tensors, not a hand-selected simplified formula.
- Report component-wise counts and whether any weights are tied/shared or excluded from trainable parameters.
- Add a programmatic count assertion/test tied to a known config. Ensure documentation formula matches implementation, or explicitly mark any formula as approximate and explain omissions.
- Never call a formula-derived estimate an observed allocation count.

**B. Context length**
- Reconcile 128 versus 64 across defaults, CLI help/examples, training, checkpoint config, and generation.
- State one canonical default only if the implementation supports it. Otherwise document per-entrypoint behavior and the migration/compatibility decision.
- Add tests for accepted maximum sequence length, overlength inputs, generation truncation, and checkpoint/config round-trip.

**C. Data-size heuristic**
- Preserve `params × 60` only as an unvalidated heuristic unless reproducible experiments establish its scope and limitations.
- Correct units and arithmetic: 810,000 × 60 = 48,600,000 bytes = 48.6 MB decimal; 117,000,000 × 60 = 7.02 billion bytes = 7.02 GB decimal.
- Do not describe this as a minimum-data law, guaranteed requirement, or evidence of model quality. Separate raw bytes, UTF-8 bytes, characters, tokens, and training examples.
- Find and correct every conflicting table/example, including claims such as "48KB minimum", "7MB+" and any incompatible parameter/data scaling statements. If the underlying empirical basis is absent, label it unvalidated and request experiment evidence rather than inventing a replacement rule.

**D. GPT-1 likeness**
- Separate architecture likeness, tokenizer/data likeness, training-recipe likeness, evaluation/capability, and engineering/tooling.
- Check every GPT-1 numeric specification against primary research sources before asserting equivalence. Distinguish GPT-1 from GPT-2 values (for example, vocabulary/tokenizer claims); do not silently substitute one model's configuration for another.
- A checkpoint/resume implementation is reproducibility/tooling infrastructure, not a direct GPT-1 likeness achievement. Full-state resume may be an inferred prerequisite for long reproducible training experiments, not proof of likeness.
- Use "GPT-1-inspired", "partial architectural comparison", or "not established" unless all stated axes have evidence. Do not use "equivalent" based on parameter count alone.

**E. Deterministic replay and resume**
- Until complete training-state resume is implemented and tested, define deterministic replay as a fresh uninterrupted run from the same initial seed/config/data/environment, subject to explicitly documented determinism limits.
- A weights-only load into a fresh optimizer is warm-start/fine-tuning, not exact continuation.
- Define full-state resume requirements: model tensors, optimizer moments and step, scheduler state, iteration, RNG state(s), sampler/data cursor/order, gradient accumulation/scaler state if used, relevant config, dataset identity/hash, and checkpoint schema/version.
- Add an interrupted-versus-uninterrupted equivalence test with controlled inputs and a stated numerical tolerance. Test checkpoint round-trip, invalid magic/version, truncation, malformed config, and backward compatibility.
- Treat the previously observed checkpoint offset concern as a hypothesis to verify at current HEAD, not as a presumed current defect.

**F. Loss trace and local evidence**
- Locate and preserve existing loss traces and OpenCode local inference/training artifacts. Link them from the canonical report with provenance, exact run/config/commit, dataset identity, logging definition, and artifact hash where available.
- Determine whether loss is per byte, per token, per sequence, or another reduction from the actual logging code. Never infer units from plotted values or filenames.
- Do not discard, overwrite, regenerate, or relabel existing participant evidence. Attribute REPORTED evidence to its producer; independently rerun only when useful and authorized.

*Claude independently confirmed the `0f889c0` specification/roadmap description and unchecked acceptance boxes, but did not read `6568e33` or the source. Its source-level non-verification is retained as a limitation of that review, not a contradiction of the source findings already recorded against the cited commit.*

*OpenCode directive from comment 6828568649 by vitrixLab user, expanded from commit 0b7bf97 findings. Applies repository-wide governance from Issues #25 and #20.*

**Relevant independent cross-review points added to Issue #40:**

- **Parameter count:** ~810,784 claimed versus ~1,228,800 in the implementation discussion; resolve from actual allocated trainable tensors and add a count assertion.
  - *From Claude cross-review (comment 6828090894):* resolve from actual allocated trainable tensors.
  - *From OpenCode directive (comment 6828568649):* report component-wise counts and whether any weights are tied/shared or excluded from trainable parameters. Add a programmatic count assertion/test tied to a known config. Ensure documentation formula matches implementation, or explicitly mark any formula as approximate and explain omissions. Never call a formula-derived estimate an observed allocation count.

- **Context:** 128 in the model description/config versus 64 in a CLI example; trace defaults and train/generation paths, then test consistency.
  - *From Claude cross-review (comment 6828090894):* trace defaults and train/generation paths.
  - *From OpenCode directive (comment 6828568649):* reconcile 128 versus 64 across defaults, CLI help/examples, training, checkpoint config, and generation. State one canonical default only if the implementation supports it. Otherwise document per-entrypoint behavior and the migration/compatibility decision. Add tests for accepted maximum sequence length, overlength inputs, generation truncation, and checkpoint/config round-trip.

- **Data-size rule:** `params × 60` gives ~48.6 MB decimal for ~810K parameters, not ~5 MB. The heuristic itself remains unvalidated and must not be described as a minimum-data law.
  - *From Claude cross-review (comment 6828090894):* the heuristic itself remains unvalidated and must not be described as a minimum-data law.
  - *From OpenCode directive (comment 6828568649):* preserve `params × 60` only as an unvalidated heuristic unless reproducible experiments establish its scope and limitations. Correct units and arithmetic: 810,000 × 60 = 48,600,000 bytes = 48.6 MB decimal; 117,000,000 × 60 = 7.02 billion bytes = 7.02 GB decimal. Do not describe this as a minimum-data law, guaranteed requirement, or evidence of model quality. Separate raw bytes, UTF-8 bytes, characters, tokens, and training examples. Find and correct every conflicting table/example, including claims such as "48KB minimum", "7MB+" and any incompatible parameter/data scaling statements. If the underlying empirical basis is absent, label it unvalidated and request experiment evidence rather than inventing a replacement rule.

- **GPT-1 likeness:** resume/checkpoint work is tooling and reproducibility infrastructure, not a direct likeness axis; full-state resume is an inferred prerequisite for long, reproducible recipe experiments.
  - *From Claude cross-review (comment 6828090894):* resume/checkpoint work is tooling and reproducibility infrastructure, not a direct likeness axis.
  - *From OpenCode directive (comment 6828568649):* separate architecture likeness, tokenizer/data likeness, training-recipe likeness, evaluation/capability, and engineering/tooling. Check every GPT-1 numeric specification against primary research sources before asserting equivalence. Distinguish GPT-1 from GPT-2 values (for example, vocabulary/tokenizer claims); do not silently substitute one model's configuration for another. A checkpoint/resume implementation is reproducibility/tooling infrastructure, not a direct GPT-1 likeness achievement. Full-state resume may be an inferred prerequisite for long reproducible training experiments, not proof of likeness. Use "GPT-1-inspired", "partial architectural comparison", or "not established" unless all stated axes have evidence. Do not use "equivalent" based on parameter count alone.

- **Replay:** until full-state resume exists, deterministic replay means a fresh uninterrupted seeded run, not weights-only restart.
  - *From Claude cross-review (comment 6828090894):* deterministic replay means a fresh uninterrupted seeded run.
  - *From OpenCode directive (comment 6828568649):* until complete training-state resume is implemented and tested, define deterministic replay as a fresh uninterrupted run from the same initial seed/config/data/environment, subject to explicitly documented determinism limits. A weights-only load into a fresh optimizer is warm-start/fine-tuning, not exact continuation. Define full-state resume requirements: model tensors, optimizer moments and step, scheduler state, iteration, RNG state(s), sampler/data cursor/order, gradient accumulation/scaler state if used, relevant config, dataset identity/hash, and checkpoint schema/version. Add an interrupted-versus-uninterrupted equivalence test with controlled inputs and a stated numerical tolerance. Test checkpoint round-trip, invalid magic/version, truncation, malformed config, and backward compatibility.

- **Loss trace:** the existing Issue #40 comments mention corpus-size/generation observations but do not provide a raw loss trace or define whether loss is per byte or per token. OpenCode should link the already-existing trace and its logging definition/run metadata; do not invent the unit or discard local results.
  - *From Claude cross-review (comment 6828090894):* link existing loss trace and its logging definition/run metadata; do not invent the unit or discard local results.
  - *From OpenCode directive (comment 6828568649):* locate and preserve existing loss traces and OpenCode local inference/training artifacts. Link them from the canonical report with provenance, exact run/config/commit, dataset identity, logging definition, and artifact hash where available. Determine whether loss is per byte, per token, per sequence, or another reduction from the actual logging code. Never infer units from plotted values or filenames. Do not discard, overwrite, regenerate, or relabel existing participant evidence. Attribute REPORTED evidence to its producer; independently rerun only when useful and authorized.
### Optimizer/sampler-state checkpoint — implementation note (2026-11-13, feature/nanolm-training-upgrade)

**Format (OBSERVED in src/nanolm.cpp):** self-describing "BNNS" v1 checkpoint persisting the Adam step, all optimizer hyperparameters (`lr`, `lr_final`, `lr_warmup_iters`, `lr_total_iters`, `b1`, `b2`, `eps`), every first- and second-moment tensor (embeddings + per-block attention/MLP), the data-sampler `std::mt19937` state (standard textual form), and the corpus token count. Loading validates magic, version, configuration (against the live model), plausibility of optimizer hyperparameters, and exact payload size (rejects truncated and trailing data); it is transactional — live model/optimizer/RNG state mutates only after full validation.

**Invocation (OBSERVED):**
- Save: `nanolm train <data.txt> [model.bin] [iters] [lr] ... [total_iters]` → writes `<model.bin>` and `<model.bin>.state`
- Resume: `nanolm resume <data.txt> <model.bin> [extra_iters] [total_iters]` → loads `<model.bin>` + `<model.bin>.state`, refuses a corpus whose token count differs from the saved sampler range, continues training, and propagates any save/load failure to a non-zero exit

**Guarantee boundary:** with an identical corpus and checkpoint configuration, an interrupted run saved and resumed is bitwise-equal (0-ULP) to an uninterrupted run — verified by `resumed_training_matches_uninterrupted` and by CLI `cmp` of uninterrupted vs interrupted+resumed checkpoints. This is **optimizer+sampler-state resume**, not a claim about any other determinism axis: the global init/generation RNG is not training state (weights load supersedes initialization), and dataset identity beyond the token count is the caller's responsibility.

**Pre-existing defect fixed (VERIFIED against 0b7bf97):** `load_weights()` sought its tensor payload to `sizeof(Config)`, but the payload starts after magic+version+Config — every weights load read all tensors 8 bytes shifted. The interrupted-vs-uninterrupted equivalence test surfaced it; the offset now matches the `save_weights()` layout, and the existing sandbox tests (round-trip via generation, rejection paths) still pass.
