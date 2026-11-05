# NanoLM Sandbox Tests — Known Scope Limitations (v2)

These limitations describe the coverage boundaries of the v1 NanoLM sandbox test
suite. They must remain visible when interpreting execution results. Do not
silently upgrade the claims made by the tests.

## 1. Attention finite differences
Attention finite-difference checks cover WQ, WK, and WV weight matrices only.
WO (output projection) and input gradients are NOT covered by finite-difference
verification in the v1 tests.

## 2. Next-token reference
The next-token full-window reference test uses probabilities derived from the
same forward cache used by the model under test. It is an internal consistency
check, not an independently computed reference implementation.

## 3. Checkpoint rejection coverage
Checkpoint rejection coverage is limited to a truncated checkpoint fixture.
Other checkpoint corruption modes (e.g., truncated header, mismatched magic,
partial weight blocks) are not exercised by the v1 tests.

## 4. CLI failure coverage
CLI failure coverage checks the save-failure path (writing to a missing parent
directory). It does NOT cover missing corpus input, short corpus behavior, or
other CLI invocation failure modes.

## Interpretation note
A PASS result from the v1 NanoLM sandbox tests confirms the above-listed
behaviors only. It does not establish correctness for WO gradients, independent
next-token reference computation, arbitrary checkpoint corruption, or missing/short
corpus CLI handling.