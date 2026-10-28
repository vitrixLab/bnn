# BNN — Binary Neural Network Experiments

**Experimental C++17 research repository for low-bit quantization, packed matrix-vector kernels, and model/runtime engineering.**

> **Research status:** prototype, not production-ready. The repository contains quantization primitives and a separate nanoLM source port. The quantized kernels are **not yet integrated into nanoLM inference**. Performance, model-quality, and end-to-end speedup claims require reproducible, workload-equivalent measurements.

## Research paper

**A Repository-Wide Mathematical and Engineering Audit of Vitrix Lab: Quantization Kernels, nanoLM Architecture, Binary-Skill Runtime, and Verification Evidence**

[Read the research-paper draft on Academia.edu](https://www.academia.edu/179755366/DRAFT_A_Repository_Wide_Mathematical_and_Engineering_Audit_of_Vitrix_Lab_Quantization_Kernels_nanoLM_Architecture_Binary_Skill_Runtime_and_Verification_Evidence)

The paper examines the repository as a collection of distinct components—not as a fully integrated quantized neural-network runtime. It discusses the quantization mathematics, nanoLM architecture, Binary-Skill binary format and benchmark evidence, verification practices, provenance, and current engineering limitations.

> **Interpretation note:** the linked work is labeled a **draft**. Treat its findings as an engineering/research audit, not as independent peer review or proof of production readiness. Repository behavior and benchmark claims should be checked against the exact source revision and reproducible test artifacts.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/weight_loss.hpp` | INT4 quantization/dequantization, INT4 × dynamically quantized INT8 matrix-vector multiplication, binary sign packing, and BIN1 XNOR/popcount-style matrix-vector multiplication. |
| `tests/weight_loss_test.cpp` | Assertion-based smoke tests for packing/decoding, finite outputs, zero weights, a partial BIN1 word, and rejection of infinite input. |
| `src/nanolm.cpp` | Separate source port of a character-level Transformer model. It is not currently driven by the quantized kernels. |
| `binary-skill/` | Standalone Binary-Skill binary-format/runtime and benchmark experiment. Evaluate its results separately from the neural-network kernels. |
| `.github/workflows/cpp-ci.yml` | C++ continuous-integration workflow and checks configured for this repository. |

The nanoLM source was ported from [`vitrixLab/blank/ai/gpt.cpp`](https://github.com/vitrixLab/blank/blob/68f24e5541b9257e6c77e77f34fc852378a29da6/ai/gpt.cpp), source revision `68f24e5541b9257e6c77e77f34fc852378a29da6`.

## Quantization methods

### INT4 weights with dynamic INT8 activations

For each weight row (r), the reference implementation derives a symmetric row scale from the maximum absolute weight:

\[
m_r = \max_c |W_{r,c}|, \qquad
s_r = \begin{cases}m_r/7,&m_r>0\\0,&m_r=0\end{cases}
\]

Nonzero rows are quantized to signed integer values in ([-7,7]), with clipping after rounding. Two 4-bit values are packed into each byte. Activations are dynamically quantized to signed INT8, and the matrix-vector result is rescaled using the weight-row and activation scales.

This is a simple reference implementation. Its numerical error, edge cases, and performance must be measured against an FP32 baseline on representative workloads.

### BIN1 weights

The BIN1 path stores one sign bit per weight and uses a per-row mean-absolute weight scale. Activation signs are packed and compared with weight signs using XNOR/popcount-style arithmetic. Partial words must be masked so padding bits do not contribute to the dot product.

The storage and arithmetic scheme is a prototype. Accuracy, overflow behavior, tail handling, and speed should be evaluated for multiple dimensions and data distributions before production use.

## Build and run the quantization smoke tests

From the repository root, using a C++17 compiler:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -pedantic tests/weight_loss_test.cpp -o /tmp/weight_loss_test
/tmp/weight_loss_test
```

Expected success output:

```text
weight_loss tests passed
```

These are assertion-based smoke tests—not a full numerical-accuracy, fuzzing, stress, sanitizer, or performance suite.

## nanoLM integration boundary

The nanoLM model source and the quantization kernels are currently separate. **Do not infer that nanoLM inference uses INT4 or BIN1 merely because both are present in this repository.** No end-to-end quantized inference, training/backpropagation modification, checkpoint conversion, model-quality evaluation, or verified end-to-end speedup is claimed by this README.

A future inference-only adapter could evaluate attention and MLP projections while retaining the FP32 path as a reference. Normalization, softmax, residual additions, training behavior, and checkpoint serialization should remain unchanged until numerical parity and integration tests support any change.

## Binary-Skill and benchmark evidence

The `binary-skill/` component is a separate experiment. Its binary-format measurements and benchmark output should not be conflated with neural-network quantization performance.

Benchmark numbers are only meaningful when the tested source revision, workload, input sizes, warm-up policy, iteration counts, aggregation method, compiler, build flags, hardware, and runtime environment are recorded. A benchmark result file that contains a manually fixed commit string is not, by itself, proof that the recorded numbers came from that exact source revision. Likewise, a benchmark operation name or metadata field is not sufficient to establish that an operating-system memory-mapping mechanism was actually used.

Do not present a speedup figure as validated unless the baseline and candidate perform equivalent work and the result can be reproduced from documented commands and revision-linked artifacts.

## Current limitations

- Quantized kernels are not wired into nanoLM inference.
- INT4 uses a symmetric per-row scale and the value range `[-7, 7]`; broader accuracy and workload testing is still needed.
- BIN1 numerical error must be compared with an FP32 reference on representative weights and activations.
- Existing quantization tests are smoke tests, not comprehensive correctness or robustness evidence.
- Large-dimension overflow, integer arithmetic, and index-boundary behavior need explicit review.
- Performance claims require equivalent workloads, a documented methodology, and revision-bound results.
- Binary-Skill benchmark/provenance findings apply to that component and must not be generalized to the quantization kernels or nanoLM.
- Passing compilation or CI checks does not by itself establish model-quality parity, production safety, or end-to-end performance.

## Verification roadmap

1. **Correctness:** add deterministic reference tests for INT4 and BIN1, including absolute/relative error bounds, zero rows, odd dimensions, extreme values, and invalid inputs.
2. **Robustness:** exercise boundary dimensions and large inputs; run AddressSanitizer and UndefinedBehaviorSanitizer where supported.
3. **CI coverage:** ensure the quantization test target is built and executed in CI on supported platforms.
4. **Integration:** add an explicit nanoLM inference adapter only behind a testable boundary, preserving the FP32 implementation as the reference.
5. **Numerical parity:** measure layer outputs and end-to-end model outputs against FP32 with declared tolerances.
6. **Performance:** benchmark equivalent operations and representative end-to-end workloads; record warm-ups, repetitions, compiler, flags, hardware, memory use, and the exact Git commit.
7. **Provenance:** generate benchmark metadata from the checked-out revision and preserve the raw samples and exact commands used to produce summaries.

## Research and engineering principle

A result is not fully substantiated by a formula, a passing smoke test, or a timing number alone. Correctness, data integrity, numerical accuracy, reproducibility, and provenance must be evaluated together:

\[
A = C \land I \land N \land R \land P
\]

where `C` means correctness, `I` means integrity, `N` means numerical accuracy, `R` means reproducibility, and `P` means provenance. The symbol `AND` means every condition must hold. This is a proposed acceptance rule for research evidence, not a claim that every gate has already passed.

## Contributing

When submitting changes:

- State which component is affected: quantization kernels, nanoLM, Binary-Skill, benchmark tooling, or CI.
- Include reproducible commands and the exact source revision for reported results.
- Separate observed results from hypotheses and planned work.
- Avoid claiming production readiness, model-quality parity, or speedup without supporting evidence.

## License

Check the repository's license file and the upstream model source's terms before redistributing or incorporating code. Do not assume that the presence of source code alone establishes a license grant.
