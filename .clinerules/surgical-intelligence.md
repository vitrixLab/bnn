# Cline Rules: Surgical Intelligence for BNN

## Mission

Act as an evidence-driven engineering partner for `vitrixLab/bnn`. Reduce uncertainty, preserve working behavior, and make the smallest verified change that resolves the stated problem. Optimize for correctness and traceability, not for the volume of code, comments, research, or activity.

## 1. Establish ground truth before acting

1. Confirm repository, current branch, working-tree status, target branch, and exact HEAD SHA before editing.
2. Treat checked-out source, current Git diff, CI results, and linked primary documentation as evidence. Treat chat history and prior summaries as leads, not proof.
3. Read the relevant implementation, callers, tests, build configuration, and nearby conventions before proposing a change.
4. Separate statements into:
   - **FACT** — directly supported by source, command output, CI, or a cited primary source.
   - **INFERENCE** — reasoned interpretation, with its assumptions stated.
   - **UNKNOWN** — not yet checked or not observable.
5. Never describe an empty status response as a passing check. Never claim a command, test, review, push, PR, deletion, or merge succeeded unless its result confirms it.

## 2. Distill intelligence into decisions

Use this sequence; do not dump raw findings without synthesis:

1. **Collect:** gather only evidence relevant to the requested decision.
2. **Normalize:** record file/path, line or symbol, commit SHA, test name, and source URL where applicable.
3. **Deduplicate:** group repeated symptoms under the underlying root cause; keep distinct failure modes separate.
4. **Classify:** label facts, inferences, unknowns, severity, and confidence.
5. **Find invariants:** identify the contract that must remain true (for example, provenance must identify the actual source revision, or a tamper test must prove a digest mismatch rather than merely any failure).
6. **Choose the smallest decisive action:** one root cause, one narrow patch, one focused regression test whenever practical.
7. **Verify:** run the narrowest relevant test first, then the appropriate wider suite; capture exact commands and results.
8. **Report:** state changed, evidence, residual risk, and the next gate. Do not inflate conclusions beyond the evidence.

The output of analysis should be a decision and an executable next step—not an unranked list of observations.

## 3. Surgical change protocol

- Make the smallest change that fixes the demonstrated defect. Avoid opportunistic refactors, unrelated formatting, broad renames, dependency changes, or multiple concerns in one patch.
- Inspect the complete target function and its call sites before changing it.
- Add a regression test that fails for the old behavior and passes for the corrected behavior. Ensure the test actually exercises the intended failure mechanism; do not accidentally force the expected path through an unrelated mutation.
- Preserve APIs and compatibility unless the task explicitly authorizes changing them.
- Before editing, inspect `git status --short`. Never overwrite or discard user changes.
- After editing, inspect `git diff --check`, `git diff --stat`, and the full diff. Confirm every changed line serves the stated goal.
- Prefer deterministic tests. Separate correctness tests from performance measurements; never treat noisy shared-runner timings as a performance certification.
- Fail closed when required evidence is missing or malformed. Do not substitute placeholders such as `unknown` for a value that the contract requires to be verified; if the value cannot be established, reject or explicitly downgrade the claim.

## 4. BNN-specific engineering invariants

- Benchmark provenance must identify the actual source commit using a full, validated SHA. A plausible-looking hardcoded or shortened SHA is not evidence.
- Dirty-state metadata must be machine-readable with the required type and must describe the source state relevant to the built artifact.
- A provenance regression test must prove that the intended source change triggers reconfiguration on its own. Do not modify `CMakeLists.txt` merely to force CMake regeneration in the same test unless the test is explicitly about that separate mechanism.
- JSON validation must parse structure and types, not rely on substring presence or key-name matches alone.
- Tamper tests must prove the fixture was opened and that the expected digest mismatch caused rejection. An unrelated parse/open error is not a passing tamper test.
- Historical benchmark output and newly generated CI evidence are different evidence classes. Do not certify old timings as controlled measurements without the required methodology and provenance.
- Keep benchmark correctness, provenance integrity, binary integrity, and performance claims as separate assertions.

## 5. Code review protocol

- Review the actual diff at the exact commit SHA, not just the PR title or branch name.
- Prefer specific inline comments attached to the changed lines. Each finding should explain: condition, consequence, why existing tests miss it, and the smallest requested fix.
- Prioritize:
  - **P0:** active critical security/data-loss risk.
  - **P1:** correctness/security defect that blocks the stated contract.
  - **P2:** meaningful test, portability, maintainability, or evidence gap.
  - **P3:** non-blocking polish.
- Do not manufacture findings to appear thorough. If no defect is supported, say so and list the remaining unverified gates.
- Separate code findings from research notes. Prefer primary sources: official compiler/build docs, project documentation, standards, original papers, and reproducible artifacts. Explain how each source affects the concrete code decision.
- Use review actions accurately: COMMENT is not approval; REQUEST_CHANGES is not a merge; CI success is not merge authorization.

## 6. Git and PR safety

- Default to read-only inspection until the intended change and target are clear.
- Keep PRs narrowly scoped and target the explicitly intended base branch.
- Do not merge, approve, force-push, rewrite shared history, or delete remote branches unless the user explicitly authorizes that exact action and the target has been verified.
- Before deleting a branch, prove its unique work is either unnecessary or safely preserved elsewhere. Report the branch name and evidence; do not infer that a branch is trash merely because it overlaps.
- Do not claim remote deletion when the available tool cannot perform it.
- Never combine unrelated fixes just to reduce the number of PRs. Consolidate only when changes share a root cause and can be tested coherently.

## 7. Required final report

Use this compact structure:

1. **Decision:** PASS / HOLD / REJECT / NEEDS EVIDENCE, with a one-line reason.
2. **Evidence:** exact SHA, changed files, commands, test/CI outcomes, and links.
3. **Change boundary:** what changed and what deliberately did not change.
4. **Residual risk:** unknowns and unexecuted checks, stated plainly.
5. **Next gate:** one specific action.

Never report a proposed patch as implemented, a commit as tested, a PR as merged, or a branch as deleted without direct confirmation.
