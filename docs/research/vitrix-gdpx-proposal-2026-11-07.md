# VITRIX-GDPx Proposal and GDPx/GDPval Source Audit

**Date:** 2026-11-07  
**Repository:** `vitrixLab/bnn`  
**Status:** Research source audit and proposal for human review  
**Scope:** GDPx (SemiAnalysis), GDPval (OpenAI), and a proposed VitrixLab engineering benchmark  
**Evidence rule:** A first-party source can establish what its publisher says; it does not by itself establish independent validation or peer review.

---

## Executive summary

GDPx is a resolvable SemiAnalysis benchmark with a first-party leaderboard at <https://gdpx.semianalysis.com/> and a first-party SemiAnalysis newsletter/technical announcement at <https://newsletter.semianalysis.com/p/gdpx-measuring-frontier-model-capabilities>. The leaderboard establishes the benchmark identity, 11 domains, domain/partner equal weighting, and a mixture of binary and continuous grading. Several additional specifications repeated in secondary coverage—206 tasks, 14 partners, an eight-hour task limit, pass@1 averaged over three attempts, native agent harnesses, and a November 2026 launch date—are retained as **REPORTED**, not upgraded to independently verified facts in this audit because the retrieved first-party page content did not establish each detail.

GDPval is independently documented by OpenAI's paper and announcement and by OpenAI's public Hugging Face dataset. The paper is titled *GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks*, by Tejal Patwardhan et al.; its arXiv submission date is 2025-10-05. OpenAI's public announcement is dated 2025-09-25. The gold subset contains 220 tasks across 44 occupations. The paper describes a full set of 1,320 tasks across 44 occupations and uses blind human expert comparison as its primary evaluation metric.

GDPx and GDPval are distinct benchmark products from different publishers. Their names and broad interest in economically relevant work should not be treated as evidence that their task sets, protocols, metrics, or scores are interchangeable.

## Evidence-status vocabulary

- **VERIFIED:** The cited primary source resolves and explicitly supports the stated fact.
- **REPORTED — not independently verified:** The claim appears in secondary coverage or an unverified summary, but this audit did not resolve sufficient primary-source content to verify it.
- **UNVERIFIABLE:** A source or primary technical record was not located or could not be resolved sufficiently. This does not prove that no such source exists.
- **REJECTED-SOURCE:** The source type or provenance fails the stated source-quality bar and must not support factual claims.
- **PROPOSED:** A VitrixLab design decision or recommendation, not an empirical finding.

---

## Layer 1 — Verified facts about GDPx and GDPval

### 1.1 GDPx — SemiAnalysis

**Official benchmark page:** <https://gdpx.semianalysis.com/>

**First-party announcement:** <https://newsletter.semianalysis.com/p/gdpx-measuring-frontier-model-capabilities>

The official leaderboard identifies GDPx as a SemiAnalysis benchmark and displays 11 of 11 domains. It states that the GDPx score averages each model across all 11 domains, with each domain weighted equally and each partner within a domain weighted equally. It also states that tasks use binary and continuous grading depending on the domain, and that the default score averages across three attempts. The latter page wording does not, in the retrieved content, establish the exact headline metric as pass@1.

Verified GDPx facts in this audit:

1. **Existence and identity:** The first-party SemiAnalysis leaderboard resolves and identifies GDPx.
2. **Domain count:** The leaderboard displays 11 of 11 domains.
3. **Grading approach:** The leaderboard describes binary and continuous grading depending on the domain.
4. **Weighting:** The leaderboard says each of the 11 domains is weighted equally and each partner within a domain is weighted equally.

Primary source: <https://gdpx.semianalysis.com/>

The SemiAnalysis newsletter is a first-party publication, but it is a newsletter/technical announcement, not a peer-reviewed paper. It may be cited as a publisher statement when the relevant passage is directly available; it must not be described as peer-reviewed scientific validation.

**GDPx technical-paper status:** No peer-reviewed GDPx paper, DOI, or independently verifiable scholarly technical report was located in this audit. Status: **UNVERIFIABLE — not located**, not a claim that such a paper cannot exist.

### 1.2 GDPval — OpenAI

**Primary paper (PDF):** <https://cdn.openai.com/pdf/d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf>  
**arXiv record:** <https://arxiv.org/abs/2510.04374>  
**OpenAI announcement:** <https://openai.com/index/gdpval/>  
**OpenAI dataset:** <https://huggingface.co/datasets/openai/gdpval>

Verified facts:

1. The paper title is *GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks*.
2. The paper lists Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, Natalie S. Kim, Patrick Chao, Samuel Miserendino, Gildas Chabot, David Li, Michael Sharman, Alexandra Barr, Amelia Glaese, and Jerry Tworek.
3. The arXiv record dates the submission to **2025-10-05**. This is distinct from the OpenAI announcement date, **2025-09-25**.
4. GDPval covers 44 occupations across nine U.S. economic sectors.
5. The paper describes a full set of 1,320 tasks and a public gold subset of 220 tasks.
6. The public dataset hosted under OpenAI's Hugging Face organization contains a train split with 220 rows, 44 occupation classes, and nine sector classes.
7. Tasks are constructed from representative work of industry professionals; the paper reports an average of 14 years of professional experience among task creators.
8. The paper identifies blind, head-to-head human expert comparison as its primary evaluation metric. The automated grader for the 220-task gold subset is described as experimental.
9. GDPval is not a fully interactive, unrestricted live-workplace deployment benchmark in its initial version. OpenAI describes the current evaluation as one-shot and identifies greater interactivity and contextual nuance as future work.

Primary sources:
- Paper: <https://cdn.openai.com/pdf/d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf>
- arXiv metadata: <https://arxiv.org/abs/2510.04374>
- OpenAI announcement: <https://openai.com/index/gdpval/>
- Dataset: <https://huggingface.co/datasets/openai/gdpval>

### 1.3 GDPx is not GDPval

GDPx is published by SemiAnalysis; GDPval is published by OpenAI. The official SemiAnalysis leaderboard and OpenAI paper identify separate benchmark products, with different publishers and documented evaluation descriptions. Their names, scores, tasks, and methodologies must not be conflated.

No GDPx score should be described as a GDPval score, and GDPval's 220-task gold subset must not be attributed to GDPx.

---

## Layer 2 — Reported claims (secondary sources only)

The following claims appeared in secondary descriptions of GDPx but were not independently established from the retrieved first-party leaderboard content. Each remains **REPORTED — not independently verified**.

| Claim | Source type encountered | Status | What is needed to promote to verified |
|---|---|---|---|
| GDPx launched in November 2026 (some secondary reports specify 2 November) | Secondary benchmark/news coverage | REPORTED — not independently verified | Dated first-party launch announcement or dated official release record |
| 11 specialized domains | Official leaderboard also confirms this | VERIFIED; not merely reported | No additional evidence needed for the domain count |
| 14 specialized data partners | Secondary benchmark coverage | REPORTED — not independently verified | First-party methodology/announcement passage enumerating or explicitly counting partners |
| 206 tasks | Secondary benchmark coverage | REPORTED — not independently verified | First-party task inventory or explicit count in an accessible SemiAnalysis methodology statement |
| Tasks can require up to eight hours | Secondary benchmark coverage | REPORTED — not independently verified | First-party task protocol specifying the time limit and whether it is universal or default |
| Binary and partial/continuous grading | Official leaderboard confirms binary and continuous grading | VERIFIED | No additional evidence needed for the general grading distinction |
| Headline metric is pass@1 across three attempts | Secondary benchmark coverage | REPORTED — not independently verified | First-party scoring specification defining pass@1, attempt aggregation, and treatment of partial credit |
| Equal weighting across domains and partners | Official leaderboard confirms this | VERIFIED | No additional evidence needed for the stated weighting rule |
| Models use native agent harnesses where available | Secondary benchmark coverage | REPORTED — not independently verified | First-party evaluation protocol naming harness-selection and fallback rules |

**Counting note:** The table preserves the original requested claims while distinguishing claims confirmed by the official leaderboard from those not confirmed in the retrieved first-party material. A first-party newsletter announcement is not peer-reviewed; a directly accessible passage from it could establish a publisher-reported specification, but not independent validation.

---

## Layer 3 — VITRIX-GDPx design proposal (PROPOSED)

This section describes a proposal by VitrixLab. It is not a research finding and does not cite external validation.

All domains, scoring weights, task definitions, gates, and roadmap elements below are **PROPOSED**. They are not official GDPx specifications and must not be represented as a replication or certification of GDPx.

### 3.1 Proposed objective

**PROPOSED:** Develop a reproducible benchmark for evaluating whether AI engineering agents can complete realistic, repository-level engineering assignments under controlled professional constraints.

The benchmark should measure deliverable quality, not merely whether an agent can produce plausible explanations or pass visible unit tests.

### 3.2 Proposed domains

| Domain | Proposed scope | Example BNN assignment |
|---|---|---|
| D1 — Kernel engineering | Correctness, numerical behavior, edge cases, and performance preservation | Diagnose and correct an INT4/BIN1 kernel defect |
| D2 — Research engineering | Reproduction, measurement integrity, and scientific reporting | Reproduce a quantization experiment with preserved raw data |
| D3 — Software engineering | Repository comprehension, minimal repair, and regression control | Resolve an inherited defect without unrelated changes |
| D4 — Systems engineering | Build systems, CI, platform compatibility, and integration | Diagnose CMake/CTest/CI failures across supported platforms |
| D5 — AI governance | Authorization, provenance, scope control, and honest evidence reporting | Execute an authorized change while respecting review and merge boundaries |

### 3.3 Proposed evaluation conditions

**PROPOSED:**

- Use a pinned repository commit and a documented environment image or reproducible setup.
- Provide realistic issue context, existing code, build configuration, and relevant documentation.
- Define the expected deliverable and acceptance criteria before the run.
- Separate public smoke tests from hidden acceptance tests.
- Record agent identity, model/version, harness, tool permissions, task start/end times, resource budget, and all relevant run identifiers.
- Preserve raw logs and test output; do not rely only on an agent-written summary.
- Use isolated branches or disposable worktrees. Do not expose production credentials or grant unrestricted access to production systems.
- Have independent domain experts review tasks and subjective deliverables.
- Repeat tasks where stochasticity matters and report the number of attempts; do not silently select only the best attempt.
- Publish limitations, exclusions, grader disagreements, and failure cases.

**PROPOSED distinction about “real industry environments”:** Realistic industry tasks do not require running experiments against an employer's live production system. A controlled, representative environment can preserve real workflows while protecting customer data, credentials, operational systems, and intellectual property. Any claim of live-production evaluation would require separate authorization, safety controls, and evidence.

### 3.4 Proposed scoring rubric

| Dimension | Proposed weight |
|---|---:|
| Functional correctness | 25% |
| Numerical accuracy and domain validity | 20% |
| Test quality and coverage | 15% |
| Reproducibility and evidence integrity | 15% |
| Engineering integration and maintainability | 10% |
| Security and governance compliance | 10% |
| Efficiency and execution cost | 5% |
| **Total** | **100%** |

**PROPOSED scoring safeguards:**

- A critical security or authorization violation may trigger a benchmark-level failure regardless of weighted score.
- Fabricated tests, fabricated CI, altered raw measurements, or unsupported claims of success trigger an evidence-integrity failure.
- A solution that passes visible tests but fails hidden acceptance tests must be scored against the hidden-test result.
- Human review must be blind to model identity where feasible for subjective quality judgments.
- Report each dimension separately; do not publish only a composite score.
- Publish task-level uncertainty and sample size. A small pilot must not be advertised as a stable ranking.

### 3.5 Proposed three-level benchmark distinction

**PROPOSED taxonomy:**

1. **Knowledge benchmark:** Does the system know or explain the relevant concept?
2. **Engineering benchmark:** Can the system implement, test, debug, and verify a solution in a controlled technical environment?
3. **Industry-work benchmark:** Can the system deliver acceptable professional work under realistic task constraints, domain standards, time/resource limits, and independent evaluation?

These levels are related but not interchangeable. Success on a knowledge test does not establish engineering execution; passing a repository task does not by itself establish broad workplace capability.

### 3.6 Proposed BNN pilot tasks

**PROPOSED pilot 1 — INT4/BIN1 kernel verification**
- Supply a pinned BNN revision, quantization kernels, build system, and existing tests.
- Ask the agent to investigate a defined correctness gap against an FP32 reference.
- Evaluate numerical error, zero rows, odd dimensions, tails, partial BIN1 words, extreme values, invalid inputs, and regressions.
- Require raw test output and an evidence report.

**PROPOSED pilot 2 — Quantization experiment reproduction**
- Supply a pinned experiment specification and approved data.
- Require reproducible commands, environment capture, preserved raw samples, and an analysis report.
- Grade whether the result can be independently reproduced from the submitted evidence.

**PROPOSED pilot 3 — Inherited repository engineering**
- Supply an issue with realistic historical context and a controlled codebase.
- Require root-cause analysis, a minimal patch, regression tests, and exact commit references.
- Grade scope discipline, preservation of unrelated behavior, test execution, and honest reporting.

### 3.7 Proposed phased roadmap

1. **Specification:** Define task templates, acceptance criteria, task security, scoring, evidence format, and exclusions.
2. **Environment:** Pin repository snapshots, dependencies, build images, tool permissions, and reset procedures.
3. **Pilot:** Create the three BNN tasks above and validate task clarity with independent engineers.
4. **Grading:** Implement deterministic hidden tests and structured expert rubrics; measure grader agreement.
5. **Comparison:** Run multiple agents under matched budgets and permissions. Preserve all attempts and report cost/time alongside quality.
6. **Publication:** Publish the benchmark protocol, task provenance, limitations, reproducibility artifacts, and results only after independent review.

### 3.8 Proposed validity and governance gates

- **Task validity:** Independent practitioners confirm that each task reflects meaningful engineering work.
- **Construct validity:** Each score dimension measures the capability it claims to measure.
- **Reliability:** Repeated runs and independent graders produce interpretable variation.
- **Contamination resistance:** Keep hidden acceptance tests and task variants from becoming training targets.
- **Reproducibility:** Pin the environment, code, dependencies, test data, and evaluation version.
- **Safety:** Use isolated environments, least-privilege credentials, and synthetic or properly authorized data.
- **External validity:** Do not generalize from BNN kernel tasks to all software engineering or all professional work.
- **Publication integrity:** Release only claims supported by preserved evidence and independent review.

---

## Connection to System 3-A

**Source document:** *Measuring Dual-Process and Tri-System Constructs in Information Technology: A Scoping Review and a Design Specification for a System 3-A Measure*, VitrixLab / Hopkins Talent Group LLC, dated 2026-11-03. The uploaded PDF is a **working draft / preliminary report**, not a peer-reviewed finding.

The draft distinguishes System 3-H (human reflective/metacognitive processing) from System 3-A (externalized AI cognition). It proposes a task-anchored questionnaire-plus-task study, including manipulated AI-advisor accuracy and confidence, baseline and post-task measures, and behavioral criteria.

This document is referenced as an internal working-draft source only. Its own abstract and limitations state that all counted empirical rows are graded unverified, the search has limitations, and the proposed study is not pilot-ready. The design proposal must not be described as validated or preregistration-ready.

**File provenance:** Uploaded conversation attachment, filename:
`DRAFT_Measuring_Dual_Process_and_Tri_System_Constructs_in_Information_Technology__A_Scoping_Review_and_a_Design_Specification_for_a_System_3_A_Measure_v3.pdf`

**Publication statement:** The draft says a preliminary report was posted to academia.edu; this audit did not independently verify the public record or exact text match. Treat that publication statement as reported by the draft, not independently verified.

---

## Rejected sources

The following source classes and entries are excluded from use as verified support.

### A. Explicitly rejected source classes

- **Korean blog summaries (including “논문 요약” pages): REJECTED-SOURCE.** Secondary summaries are not primary research.
- **SusVibes Leaderboard: REJECTED-SOURCE.** No qualifying primary technical report was established.
- **“NUS - Wei Li is Hiring”: REJECTED-SOURCE.** Recruitment page, not research evidence.
- **Any `images.openai.com/static-rsc-4/*` URL: REJECTED-SOURCE.** Image thumbnail, not a documentary source.
- **Undated or unattributed newsletter passages: REJECTED-SOURCE.** Publication provenance and date cannot be established.
- **BenchLeader GDPx summary: REJECTED-SOURCE for primary factual support.** Secondary leaderboard interpretation.
- **ArtificialWatch GDPx summary: REJECTED-SOURCE for primary factual support.** Secondary reporting.
- **Evals Review GDPx summary: REJECTED-SOURCE for primary factual support.** Secondary benchmark description.

### B. System 3-A draft bibliography entries that are unverified or fail provenance requirements

These entries are not silently deleted from the source draft. They are listed here because the draft itself labels them unverified, secondary-only, unretrieved, or lacking a primary citation. They must not be used as verified support until resolved.

| Draft citation | Source-quality disposition |
|---|---|
| GS-3 Working Group (2025), *Generative System 3* | UNVERIFIED — draft's description may be internally consistent, but the underlying indices/source are unverified |
| Goh et al. (2025), *The Generative AI Dependency Scale (GAIDS)* | UNVERIFIED — venue unverified; draft labels it secondary-only |
| Goddard et al. (2019), *Automation bias* | UNVERIFIED — venue and exact bibliographic record not established in draft |
| González et al. (2026), *AI-Based Cognitive Offloading (AIBCO)* | UNVERIFIED — venue unverified; draft labels it secondary-only |
| PTRR Working Group (2026), *PTRR = Prompts, Time, Results, Rubric* | UNVERIFIED — primary citation required |
| Quad-Process Theory Working Group (2025), *Quad-Process Theory* | NOT RETRIEVED |
| Shaw and Nave (2026), *Tri-System Theory and cognitive surrender* | UNVERIFIED — cited as an SSRN/OSF preprint without a resolved primary URL in the draft |
| Zhu et al. (2026), *Dependent-Autonomous Cognitive Offloading Scale (DACOS)* | UNVERIFIED — draft labels it secondary |
| Recent software-engineering dual-process survey | TODO — citation not identified |
| Security dual-process review | TODO — citation not identified |
| HCI/IS dual-process review | TODO — citation not identified |
| ML System 1/System 2 survey | TODO — citation not identified |

### C. Recalled bibliography entries not verified by this GDPx/GDPval audit

The System 3-A draft marks the following citations **RECALLED — verify**. They are not rejected as inherently invalid works; they are excluded from verified evidentiary support in this file because this audit did not individually resolve their primary records and identifiers:

- Cacioppo and Petty (1982), *The Need for Cognition*.
- Egelman and Peer (2015), *Scaling the Security Wall: Developing a Security Behavior Intentions Scale (SeBIS)*.
- Evans and Stanovich (2013), *Dual-process theories of higher cognition: Advancing the debate*.
- Hodgkinson and Sadler-Smith (2003), *Complex or unitary? A critique and empirical re-assessment of the Allinson–Hayes Cognitive Style Index*.
- Jian, Bisantz, and Drury (1998), *Foundations for an empirically determined scale of trust in automated systems*.
- Kahneman (2011), *Thinking, Fast and Slow*.
- Körber (2018), *Theoretical considerations and development of a questionnaire to measure trust in automation*.
- Larue et al. (2013), *Computational implementation of the tripartite model*.
- Lee and See (2004), *Trust in automation: Designing for appropriate reliance*.
- Logg, Minson, and Moore (2019), *Algorithm appreciation: People prefer algorithmic to human judgment*.
- Luo et al. (2013), *Investigating phishing victimization with the Heuristic–Systematic Model*.
- Madsen and Gregor (2000), *Measuring human-computer trust*.
- Mohanani et al. (2018), *Cognitive biases in software engineering: A systematic mapping study*.
- Pacini and Epstein (1999), *The relation of rational and experiential information processing to personality, basic beliefs, and the ratio-bias phenomenon*.
- Parasuraman and Manzey (2010), *Complacency and bias in human use of automation: An attentional integration*.
- Parsons et al. (2014), *Determining employee awareness using the Human Aspects of Information Security Questionnaire (HAISQ)*.
- Peters et al. (2020), *Updated methodological guidance for the conduct of scoping reviews*.
- Risko and Gilbert (2016), *Cognitive offloading*.
- Stanovich (2009), *Decision Making and Rationality in the Modern World*.
- Stanovich (2011), *Rationality and the Reflective Mind*.
- Thomson and Oppenheimer (2016), *Investigating an alternate form of the Cognitive Reflection Test*.
- Toplak, West, and Stanovich (2014), *Assessing miserly information processing: An expansion of the Cognitive Reflection Test*.
- Tricco et al. (2018), *PRISMA Extension for Scoping Reviews (PRISMA-ScR): Checklist and explanation*.
- Vishwanath et al. (2011), *Why do people get phished? Testing individual differences in phishing vulnerability within an integrated, information processing model*.
- Yaniv (2004), *Receiving other people’s advice: Influence and benefit*.

These are **UNVERIFIED IN THIS AUDIT**, not claims that the works do not exist. A separate reference-verification pass should resolve publisher records, DOI/ISBN/arXiv identifiers, and the specific claims each source is intended to support.

---

## Verification status summary

### GDPx/GDPval factual-claim audit

Counting convention: one row per distinct factual claim/check in the Phase 1 audit. Source-class exclusions are counted separately from factual claims.

| Status | Count |
|---|---:|
| VERIFIED | 15 |
| REPORTED — not independently verified | 5 |
| UNVERIFIABLE | 1 |
| REJECTED-SOURCE classes | 8 |

**Interpretation:** The 21 factual claims/checks comprise 15 verified, five reported-only, and one unverifiable. Eight source classes/items are rejected for primary factual support. The System 3-A bibliography's unverified and recalled entries are separately enumerated above and are not included in these GDPx/GDPval claim counts.

### Key unresolved items

1. GDPx peer-reviewed technical paper or DOI: not located.
2. GDPx launch date: reported in secondary coverage; first-party dated confirmation not established here.
3. GDPx 14-partner count and 206-task count: reported; require accessible first-party methodology text.
4. GDPx eight-hour limit, exact pass@1 aggregation, and native-harness policy: reported; require first-party protocol confirmation.
5. System 3-A draft bibliography: recalled and secondary-only references require a separate primary-source verification pass.
6. Academia.edu publication and exact-text match for the System 3-A draft: not independently verified here.

---

## Review and publication limitations

- This file is a source audit and a proposal, not a peer-reviewed research paper.
- GDPx facts are limited to what the official leaderboard and accessible first-party announcement establish.
- Reported-only GDPx specifications must not be repeated as verified facts.
- GDPval's paper and dataset establish a professional-task benchmark, but the initial GDPval evaluation is described by OpenAI as one-shot; it should not be misrepresented as unrestricted live production testing.
- The VITRIX-GDPx scoring weights and pilot tasks are design proposals and have not been validated.
- No benchmark tasks were executed, no models were evaluated, and no performance results were generated in preparing this document.
- This document does not authorize production access, publication of private data, merging, or any external research claim beyond the evidence stated above.
