# Research note: *The Algebraic Mind* (2001) as a lens for fine-tuning binarized (BIN1) nanoLM

**Tier:** nanoLM only. SLM / MLM / LLM are deferred.
**Status:** PLANNING / NOT VERIFIED. No build, test, training run or measurement was executed for this note.
**Parent:** [Issue #27](https://github.com/vitrixLab/bnn/issues/27) owns the nanoLM baseline, dataset and training plan. This note links to it and does not restate it (DRY rule).
**Drafted:** 2026-11-03

**Status labels used below**

| Label | Meaning |
|---|---|
| VERIFIED | Checked directly, with a source named |
| REPORTED | Taken from an earlier audit or issue text; NOT INDEPENDENTLY VERIFIED |
| UNKNOWN | Not established; must be read or measured first |
| NOT RUN | A command or experiment that was not executed |
| PROPOSAL | A design choice that needs authorization before it is acted on |

---

## 0. What was and was not checked when this note was written

- **VERIFIED (web catalogue records):** Marcus, G. F. (2001). *The Algebraic Mind: Integrating Connectionism and Cognitive Science.* Cambridge, MA: MIT Press (Bradford Books). Paperback edition 2003. Library and publisher records agree on author, title, publisher and year.
- **VERIFIED (fetched 2026-11-03):** the current text of Issue #27, including its scope, its "do not extrapolate from 14K" warning and its DRY rule.
- **REPORTED:** every statement about nanoLM and BIN1 in section 4. They come from an earlier audit at commit `4628303831a1233aee5a81db8614ddb73633a6b0`. That session had a repository checkout; this one has none, so none of it was re-run.
- **UNKNOWN:** what the book says at chapter and page level. The book text was not consulted. Section 2 paraphrases its widely known arguments from general knowledge. Someone holding a copy must check it before any page-cited claim is added.
- **NOT RUN:** build, tests, training, benchmarks, kernel parity, probe generation.

---

## 1. Purpose and scope

Use the book as a **conceptual source for evaluation design and hypotheses**. It is not training data.

Questions for the nanoLM tier:

1. **Q1.** After binarization (post-training and binarization-aware), what happens to rule-like generalization, compared with what happens to in-distribution bits-per-character (BPC)?
2. **Q2.** Does binarization-aware fine-tuning recover BPC? Does it recover rule generalization? Are the two decoupled?
3. **Q3.** Does the FP32 14K model show any rule generalization at all? If it does not, quantization deltas on that probe carry no information (a floor effect).

**Non-goals**

- Do not train on book text (copyright, and a roughly 14K-parameter model cannot use it).
- Do not describe the model as "algebraic" or "symbolic" unless the gates in section 8 support it.
- Do not touch SLM / MLM / LLM planning.
- Do not alter INT4/INT8 rounding work (#21, #22, #24, PR #15, PR #23) or the preservation-restricted PR #8.
- Do not publish any dataset, checkpoint or probe set before the review gate in #27.

---

## 2. What the book argues (paraphrase; UNKNOWN against the text)

Marcus argues that cognition needs ingredients that the plain multilayer-perceptron connectionism of that era did not supply by default, and that neural systems can be organized to implement symbol manipulation. The three ingredients most often associated with the book:

1. **Operations over variables.** A rule applies to any member of a class ("for any x ..."), including items never seen in training.
2. **Structured representations.** Compositional and recursive structure, with roles bound to fillers.
3. **Individuals versus kinds.** A specific token is represented separately from its category.

Arguments commonly associated with it:

- **Training independence.** In a network where each input unit's weights are updated only by its own training examples, a mapping learned for trained inputs need not extend to inputs never trained. Such a network generalizes inside its "training space" but not outside it.
- **Evidence from language and development.** Regular versus irregular inflection, and rule learning in infants.
- **A hybrid stance.** Neurally realized symbol manipulation, rather than either pure symbolic systems or eliminative connectionism.

**Caveat.** The book is from 2001. It targets multilayer perceptrons and simple recurrent networks. It predates transformers and binarized networks. Applying its claims to a transformer, or to BIN1, is a hypothesis for this project to test. It is not an inherited result.

---

## 3. Mapping the book to nanoLM probes (our interpretation, not the book's)

| Book concept | nanoLM probe (PROPOSAL) | Out-of-distribution axis |
|---|---|---|
| Operations over variables, training independence | **P1 copy / identity:** output a short input string unchanged | Symbols unseen in the probe's training distribution; symbols reserved from the corpus entirely |
| Rule applied to novel items | **P2 reduplication:** ABA vs ABB style patterns over novel symbols | Novel symbol sets; longer patterns |
| Rule versus memorization (inflection) | **P3 nonce inflection:** apply a regular rewrite rule to made-up stems, alongside memorized exceptions | Novel stems; stem length |
| Structured / recursive representation | **P4 nesting:** bracket matching, a^n b^n style strings | Depth and length beyond the training range |
| Individuals versus kinds (loose analogue) | **P5 binding:** assign values to names, then recall one | Number of bindings; reused names with new values |

**Reserved-symbol design.** The #27 prerequisite is a pinned 65-symbol vocabulary. Reserve K symbols that never appear anywhere in the training corpus. Their embedding rows then never receive a task gradient, which is the closest nanoLM analogue to the book's untrained-input-unit scenario. This is a design choice. The book does not specify it.

---

## 4. BIN1 and nanoLM facts relied on (all REPORTED, from the earlier audit)

- **REPORTED:** on random Gaussian weights (not nanoLM weights), relative error versus FP32 was about 1% for INT8, 9-14% for INT4 and 66-78% for BIN1. Cosine similarity for BIN1 was 0.62-0.75.
- **REPORTED:** BIN1 is post-training sign binarization with no binarization-aware training.
- **REPORTED:** `src/nanolm.cpp` has zero references to `weight_loss`. Quantized inference is not integrated into nanoLM.
- **REPORTED:** nanoLM is a character-level transformer (n_embd 16, n_head 4, n_layer 4, block_size 55, tied lm_head), trained with Adam at constant lr 3e-3, batch size 1. About 14,208 parameters only when V = 65.
- **REPORTED:** single-thread throughput of about 15.6K tokens/s in a sandbox at -O2. Not the project's hardware.
- **UNKNOWN:** BIN1's on-disk layout, scale granularity (per-tensor, per-row or per-block), whether the scale is the optimal one (for example mean of absolute weights), and how bits are packed. **Read the kernel source before designing the fine-tuning arms.**
- **UNKNOWN:** whether the 66-78% error on Gaussian weights carries over to trained nanoLM weights.
- **Interpretation of "Binaries":** this note assumes it means **BIN1-binarized nanoLM weights**. `binary-skill/` is reported as a separate knowledge-base file format, not a neural-network component, and is out of scope unless the owner says otherwise.

---

## 5. Fine-tuning arms (PROPOSAL)

| Arm | Description |
|---|---|
| A0 | FP32 baseline from #27 (reproduced, not assumed) |
| A1 | INT8 post-training quantization, evaluated by quantize/dequantize round trip |
| A2 | INT4 post-training quantization, same method |
| A3 | BIN1 post-training quantization, same method, using the existing kernel's numerics |
| A4 | BIN1 binarization-aware fine-tuning from the A0 checkpoint |
| A5 | A4 plus logit distillation from A0 as teacher |
| A4-S | A4 trained from scratch (ablation only) |

**A4 mechanics (PROPOSAL).** Keep latent FP32 weights. The forward pass uses `sign(w) * alpha`. The backward pass uses a straight-through estimator, passing the gradient where `|w| <= 1` and zeroing it elsewhere. Adam updates the latent weights, and latent weights are clipped to [-1, 1]. `alpha` follows whatever scale BIN1 actually stores (see section 4, UNKNOWN).

**A5 mechanics (PROPOSAL).** Loss is cross-entropy plus a weighted KL term between teacher and student logits at a temperature. Teacher and student share the pinned vocabulary.

**Ablations, one factor at a time**

- Which tensors are binarized: all linear layers, attention only, or MLP only.
- Embeddings (`wte`, `wpe`), layer norms and biases kept FP32 versus quantized.
- Scale choice: fixed versus learned.
- Learning rate and gradient accumulation. The reported baseline uses batch size 1, which is noisy under a straight-through estimator.
- Fine-tuning token budget.

**Selection rule.** Hyperparameters are chosen on the held-out BPC split only. Probe test sets are never used for tuning.

**Implementation note.** The nanoLM training loop is C++. Binarization-aware training would be implemented there. Whether the fine-tuning also needs a real quantized forward path, or fake-quantization is enough at this tier, is an open question (section 11).

---

## 6. Probe suite specification (PROPOSAL)

- Probes are generated programmatically with a fixed seed and a hashed manifest.
- Every probe has a trivial symbolic oracle that scores 100%. This proves the task is well-posed.
- Strings are embedded in the character stream as short lines with a probe tag, so nanoLM can train and evaluate on them as ordinary text.
- Splits per probe:
  - **ID:** same distribution as probe training data.
  - **Near-OOD:** unseen combinations of seen symbols.
  - **Far-OOD:** longer or deeper than anything in training.
  - **Reserved-symbol:** symbols that never appear in the corpus.
- Metric: exact-match accuracy per item. Report mean and standard deviation across seeds, the item count, and the chance level for each probe.
- Probe test sets are never in the training corpus, the teacher feedback loop or hyperparameter selection. Check overlap at a declared n-gram window, consistent with #27 gate G2.

**Floor rule.** If A0 scores at or near chance on a probe's OOD splits, that probe is marked FLOOR for this tier. Quantization deltas on a FLOOR probe are reported as uninformative. They are not evidence for or against any hypothesis.

---

## 7. Experiment matrix and compute (PROPOSAL, estimate only)

Post-training arms (A1-A3) need evaluation only, no training.

Training arms: A4, A5 and A4-S, three seeds each, so 9 runs. Wall time uses the throughput REPORTED in #27 (15.6K tokens/s, single thread, sandbox, not re-measured) and assumes **overhead multiplier m = 1**, which makes these lower bounds.

| Fine-tuning tokens per run | Per-run wall time at m = 1 | 9 runs at m = 1 |
|---|---|---|
| 1M | about 1.1 min | about 10 min |
| 4M | about 4.3 min | about 38 min |
| 16M | about 17 min | about 2.6 h |

**UNKNOWN:** the real multiplier. Latent weights, gradient masks and the A5 teacher forward pass all add cost. Measure it on the target hardware before quoting a schedule. No GPU or paid compute is assumed at this tier.

---

## 8. Gates (nanoLM only; passing them authorizes nothing larger)

| Gate | Condition |
|---|---|
| **BG0** | #27 prerequisites are done: pinned vocabulary, held-out BPC eval mode, run logging (code SHA, config, seed, vocab hash, dataset hash). Otherwise BLOCKED. |
| **BG1** | **Parity:** the quantize/dequantize path reproduces the existing kernel's numerics on identical weights, to a stated tolerance. |
| **BG2** | A0 reproduces across seeds; floor-rule outcome recorded for every probe. |
| **BG3** | Post-training degradation table (BPC and every non-FLOOR probe) for A1-A3, three seeds. |
| **BG4** | A4 beats A3 in held-out BPC by more than seed standard deviation. |
| **BG5** | A4 and A5 are reported against the char n-gram control from #27 (gate G1). |
| **BG6** | Probe conclusions limited to measured differences with uncertainty. The words "symbolic" or "algebraic" are not used for the model unless the oracle gap and the controls support them. |
| **BG7** | No publication of probes, datasets or checkpoints before the license, privacy and safety review in #27. |

---

## 9. Threats to validity

- **Floor effects.** A roughly 14K-parameter model may fail every OOD probe at every precision.
- **Capacity.** At this size, binarization may reduce capacity so far that any probe change reflects capacity, not rule structure.
- **Vocabulary dependence.** Parameter count and checkpoint layout change with the corpus until the vocabulary is pinned (#27).
- **Probe leakage.** Probe-style lines in the training corpus invalidate OOD claims.
- **Tuning on the test.** Selecting hyperparameters on probe results inflates them.
- **STE bias.** The straight-through gradient is a biased estimate. A4 gains may depend on schedule more than on binarization itself.
- **Era mismatch.** The book's critiques target 2001-era architectures.
- **Two-sided hypothesis.** Binary weights could bias toward discrete, symbol-like circuits, or they could remove the fine-grained interpolation that generalization needs. The book does not address this because binarized networks postdate it. This project should treat both directions as open.

---

## 10. References

**Copyright handling.** Do not commit the book, a scan or extended excerpts to the repository. Notes should be in the author's own words, with any quotation kept very short and page-cited.

| Reference | Status |
|---|---|
| Marcus, G. F. (2001). *The Algebraic Mind: Integrating Connectionism and Cognitive Science.* MIT Press. Hardcover ISBN: 9780262133791; [official MIT Press record](https://mitpress.mit.edu/9780262632683/the-algebraic-mind/). | **VERIFIED — publisher metadata only**; book text/chapter/page claims remain UNKNOWN until directly checked |
| Marcus, G. F., Vijayan, S., Bandi Rao, S., & Vishton, P. M. (1999). “Rule learning by seven-month-old infants.” *Science*, 283(5398), 77–80. DOI: [10.1126/science.283.5398.77](https://doi.org/10.1126/science.283.5398.77). | **VERIFIED — bibliographic record only**; empirical infant result, not evidence about nanoLM |
| Courbariaux, M., Bengio, Y., & David, J.-P. (2015). “BinaryConnect: Training Deep Neural Networks with binary weights during propagations.” arXiv:1511.00363. [arXiv](https://arxiv.org/abs/1511.00363). | **VERIFIED — preprint record**; results remain task-specific empirical findings |
| Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). “Binarized Neural Networks.” *Advances in Neural Information Processing Systems 29*. [Official NeurIPS proceedings](https://proceedings.neurips.cc/paper/2016/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html). | **VERIFIED — peer-reviewed proceedings**; reported 7× result is specific to the paper’s GPU-kernel comparison |
| Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). “XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.” arXiv:1603.05279. [arXiv](https://arxiv.org/abs/1603.05279). | **VERIFIED — preprint record**; results concern binary convolution/image-classification workloads |
| Bengio, Y., Léonard, N., & Courville, A. (2013). “Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.” arXiv:1308.3432. [arXiv](https://arxiv.org/abs/1308.3432). | **VERIFIED — preprint record**; STE is described as a heuristic estimator, not the exact derivative of a hard sign function |
| Hinton, Vinyals, & Dean (2015). “Distilling the Knowledge in a Neural Network.” | **[UNVERIFIED — DO NOT CITE]** — no acceptable stable source verified in this review |
| Wang et al. (2023), BitNet; Ma et al. (2024), BitNet b1.58. | **[UNVERIFIED — DO NOT CITE]** — exact titles, authors and stable identifiers not verified here |
| Lake, B., & Baroni, M. (2018). “Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks.” *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 2873–2882. [Official PMLR proceedings](https://proceedings.mlr.press/v80/lake18a.html). | **VERIFIED — peer-reviewed proceedings**; empirical claims limited to the reported RNN/SCAN experiments |
| Fodor and Pylyshyn (1988). “Connectionism and cognitive architecture.” | **[UNVERIFIED — DO NOT CITE]** — exact publication record and stable identifier not verified here |

---

### Citation and claim boundary (updated 2026-11-04)

Bibliographic verification establishes a publication record, not the correctness of every interpretation or its transfer to nanoLM. The Marcus book's chapter/page-level arguments remain **UNKNOWN against the text** until checked against a copy. The 1999 infant paper is an empirical result in its own experimental setting, not evidence that nanoLM exhibits rule-like generalization. BinaryConnect, BNN, XNOR-Net and STE do not demonstrate nanoLM speed, quality or generalization. The BNN 7× result is a setup-specific GPU-kernel comparison. XNOR-Net reports convolutional/image-classification results; do not rewrite its 58× convolution-operation speed result as a count of “fewer high-precision operations.”

**Classification rule:** VERIFIED means the bibliographic record was checked; it does not mean nanoLM applicability is verified. The STE is a heuristic gradient estimator, not the exact derivative of sign. Rows marked **[UNVERIFIED — DO NOT CITE]** remain excluded until acceptable source metadata is verified.

## 11. Open questions for the owner

1. Does "Binaries" mean BIN1-binarized nanoLM weights, as assumed here, or something else?
2. Is fake-quantization (quantize/dequantize inside the FP32 loop) sufficient for this tier, or must a real BIN1 forward path be added to nanoLM first?
3. Who holds a copy of the book to check section 2 against the text and add page references?
4. Where in the repository should research notes live? The suggested path is `docs/research/`; the actual docs layout is UNKNOWN.

**Authorization requested by this note:** none. It is a plan. Any code, probe generation, training run or repository change needs separate authorization.