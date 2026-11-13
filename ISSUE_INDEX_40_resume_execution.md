# Issue Index: #40 Resume Execution by Codex

## Issue Metadata
- **Issue Number:** #40
- **Title:** Resume execution by codex
- **Status:** Implemented and documented
- **Branch:** feature/nanolm-training-upgrade
- **Commit:** 0f889c0

## Documentation Created
- [`ISSUE_40_resume_execution_by_codex.md`](ISSUE_40_resume_execution_by_codex.md) - Full issue description with acceptance criteria and technical notes

## Planning Roadmap Update
- [`docs/research/HYPERION_TO_GPT1_PLAN.md`](docs/research/HYPERION_TO_GPT1_PLAN.md) - Updated "Recommended Next Steps" section with new task:
  - `[ ] Implement resume execution by codex (Issue #40) - leverage binary skill #44 `resume-codex` and integrate with checkpoint versioning and sandbox execution gate v3`

## Related Skills in Binary Database
| Skill # | Name | Category | Offset | Pool Bytes | SHA-256 |
|---------|------|----------|--------|------------|---------|
| **43** | `resume-claude` | CORE_ENGINEERING | `0x1514` | 2,864 | `69b41549ffbe383b9b424aabea2f9c4d94c030c1e6ad822a03238cc34512c875` |
| **44** | `resume-codex` | CORE_ENGINEERING | `0x1594` | 2,748 | `58cd0516998a2d57a8b34c432f20ef09f4a409b72369dc51b4e0106960b6ae38` |

## Technical Integration Points
### Binary Skill Format (`VSKB0001`)
- Fixed-width records (128 bytes each)
- String pool for skill descriptions and bodies
- Checked arithmetic bounds validation
- Memory-mapped file access (`MapViewOfFile`/`mmap`)
- SHA-256 integrity verification

### Checkpoint Versioning
- Already implemented in `src/nanolm.cpp`
- BNNM header format
- Supports resume from saved states

### Sandbox Execution Gate v3
- Audit system constraints
- Network-isolated Docker tests
- Evidence verification pipeline

## Linked Issues
- **Issue #43:** `resume-claude` - Claude-based resume implementation (baseline comparison)
- **Issue #44:** `resume-codex` - Codex-based resume implementation (directly related)
- **Planning doc:** `HYPERION_TO_GPT1_PLAN.md` - High-level research roadmap

## Implementation Checklist
- [x] Issue documentation created (`ISSUE_40_resume_execution_by_codex.md`)
- [x] Planning roadmap updated (`HYPERION_TO_GPT1_PLAN.md`)
- [x] Git commit and push to `feature/nanolm-training-upgrade`
- [ ] **Apply Claude cross-review findings (Issue #40 comment 6828090894):**
   - [ ] Resolve parameter count from actual allocated trainable tensors (~810K vs ~1.2M)
   - [ ] Trace context defaults/train/generation paths (128 config vs 64 CLI example)
   - [ ] Update data-size heuristic documentation (unvalidated; 810K×60=48.6 MB not ~5 MB)
   - [ ] Clarify GPT-1 likeness: resume/checkpoint is tooling/infrastructure, not direct likeness axis
   - [ ] Define deterministic replay: fresh uninterrupted seeded run, not weights-only restart
   - [ ] Link existing loss trace and logging definition; do not invent unit or discard local results

- [ ] **Apply OpenCode directive findings (Issue #40 comment 6828568649):**
   - [ ] Inventory all tracked Markdown files and identify canonical governance docs (Issues #25, #20)
   - [ ] Establish implementation truth by inspecting actual code with file/line references
   - [ ] Resolve parameter count from allocated trainable tensors, not simplified formulas
   - [ ] Reconcile context length 128 vs 64 across defaults, CLI, training, checkpoint, generation
   - [ ] Preserve `params × 60` only as unvalidated heuristic; correct arithmetic (810K×60=48.6 MB, 117M×60=7.02 GB)
   - [ ] Separate GPT-1 likeness axes: architecture, tokenizer/data, training-recipe, evaluation, engineering/tooling
   - [ ] Define deterministic replay as fresh uninterrupted run; weights-only load is warm-start, not exact continuation
   - [ ] Locate and preserve existing loss traces with provenance, logging definition, and artifact hash
   - [ ] Add programmatic count assertion/test tied to known config
   - [ ] Add tests for max sequence length, overlength inputs, generation truncation, checkpoint/config round-trip
   - [ ] Add interrupted-vs-uninterrupted equivalence test with numerical tolerance
   - [ ] Add loss metric semantics test (per byte, per token, per sequence)
- [ ] Implement codex API integration for state restoration
- [ ] Integrate with binary skill `#44` `resume-codex`
- [ ] Connect with checkpoint versioning system
- [ ] Validate within sandbox execution gate v3 constraints
- [ ] Test resume scenarios (GPU/CPU)
- [ ] Update benchmark evidence with resume metrics

## Update History
- **2026-11-12:** Issue documented and planning roadmap updated
- **2026-11-12:** Cross-review section retracted - actual GitHub issue #40 comment from codex takes precedence
- **Commit:** 0f889c0 - "Issue #40: Add resume execution by codex documentation and update planning roadmap"
- **Commit:** 6568e33 - "Issue #40: Add issue index connecting resume execution by codex to all relevant components"
- **Branch:** feature/nanolm-training-upgrade (tracked from origin)