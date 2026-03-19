# Vibe Coding Prompt Log

**Project:** LLM-Based Toxic Comment Classification

---

## Table of Contents

1. [Statement of AI Use](#1-statement-of-ai-use)
2. [AI Tools Used](#2-ai-tools-used)
3. [Prompt Iteration Log: Cross-Reference](#3-prompt-iteration-log-how-the-classification-prompts-evolved)
4. [Structured Prompt Log](#4-structured-prompt-log)
5. [Final Results Summary](#5-final-results-summary)

---

## 1. Statement of AI Use

AI assistance was used across more than 15 individual conversations spread across team members, covering all four GPT-4.1 pipeline methods. Rather than submitting raw chat transcripts, which span thousands of lines across multiple sessions, this document presents each substantive prompt as a structured markdown in Section 4, recording the purpose, full prompt text, model response summary, and the decision that followed. Section 3 cross-references `prompt_templates.md` Section 7, which is the authoritative record of what changed between methods and why. All sessions were conducted between 4 February and 16 March 2026.

---

## 2. AI Tools Used

| Tool | Version | Primary Use |
|---|---|---|
| ChatGPT | GPT-4.1 (OpenAI) | Method 2 few-shot examples and prompt design; RAG architecture discussions; prompt iteration for `severe_toxic` and `threat` labels; evaluation format specification |
| Claude | Sonnet (Anthropic) | Method 1 zero-shot inference engine; Method 3 RAG pipeline; Method 4 refinements; debugging parse errors and f-string issues; threshold tuning code |

---

## 3. Prompt Iteration Log: How the Classification Prompts Evolved

The full iteration log, covering the design decision made at each stage (M1→M4), the problem that motivated it, and its measured effect on performance, is documented in `prompt_templates.md`, Section 7 (Design Rationale and Iteration Log), which is the authoritative reference for this material. The development decisions that drove each iteration are also traceable through the Purpose and Decision fields of the structured prompt entries in Section 4 of this document (P-01 through P-12).

---

## 4. Structured Prompt Log

This section documents every substantive prompt used during development as a structured engineering artefact. Each entry records the prompt's **purpose**, the **full prompt text** as sent to the model, the **model response summary**, and the **decision or outcome** that followed. Ad-hoc debugging turns (syntax errors, pip install issues, file path corrections) are excluded; only prompts that produced a design decision or a code component that persists in the final submission are included.

Prompts are grouped by development phase and numbered sequentially (P-01 through P-12) for cross-referencing.

---

### Phase 1: Project Framing and Output Format Design

---

#### P-01 · ChatGPT · Initial system design and output format

**Purpose:** Establish the classification task structure, output format, and evaluation framework before any code was written.

**Prompt:**

```
You are an expert in NLP and LLM-based classification systems.

We are building a multi-label toxic comment classifier on the Jigsaw dataset.
The six labels are: toxic, severe_toxic, obscene, threat, insult, identity_hate.

Task:
1. Recommend whether to use zero-shot or few-shot prompting as the primary
   strategy, with justification.
2. Design an initial zero-shot prompt template for this six-label classification
   task.
3. Specify the output format that best supports automated evaluation
   (per-label Precision, Recall, F1, Hamming Loss, Exact Match).
4. List the evaluation metrics we should report and why each is appropriate
   for an imbalanced multi-label setting.
```

**Response summary:** Recommended strict JSON output schema over free-text labels. Proposed per-label Precision/Recall/F1, Micro-F1, Macro-F1, Hamming Loss, and Exact Match as the evaluation suite. Initial zero-shot template used a single-line output format: `NON_TOXIC or TOXIC;<TYPE_1>,<TYPE_2>`.

**Decision:** The team identified that the single-line label format would make threshold tuning and ROC-AUC computation impossible. Output format was changed to JSON confidence scores (0–100) for all labels (Methods 2–4), enabling per-label threshold calibration and ROC-AUC reporting. This pivot drove the architecture of all four methods.

---

#### P-02 · ChatGPT · Zero-shot prompt component design

**Purpose:** Define the specific components that make a zero-shot prompt effective for a 41-combination multi-label dataset, and produce a structured template.

**Prompt:**

```
You are an expert prompt engineer.

I am building a zero-shot prompt for GPT-4.1 to perform multi-label toxicity
classification on the Jigsaw dataset. The six labels are:
toxic, severe_toxic, obscene, threat, insult, identity_hate.

The dataset has 41 unique label combinations. The model must output a JSON
object with a confidence score (0–100) for each label.

Task:
1. List the essential components of an effective zero-shot prompt for this
   task, with a justification for each component.
2. Identify the three most common label confusion errors in toxicity
   classification and write decision guidelines that address each.
3. Produce a complete, production-ready zero-shot prompt template.
   Use temperature=0 and seed=42 for reproducibility.
```

**Response summary:** Identified five components, task definition, label schema with decision boundaries, output format constraint, decision rules, and deterministic settings. Addressed `insult` vs `identity_hate` confusion and flagged `threat` over-prediction from angry language. Produced the structured template that became Method 1 (full template in `prompt_templates.md`, Section 1).

**Decision:** Template adopted as the Method 1 baseline. Decision guidelines were carried forward into Methods 2–4 with progressive refinement.

---

### Phase 2: RAG Architecture and Retrieval Design

---

#### P-03 · ChatGPT · Retrieval-augmented few-shot architecture

**Purpose:** Design the RAG retrieval mechanism for dynamic few-shot example selection across 41 label combinations.

**Prompt:**

```
You are an expert prompt engineer and NLP architect.

I need to build a Retrieval-Augmented Few-Shot (RAFS) system for multi-label
toxicity classification with 41 unique label combinations.

The few-shot pool has ~151,000 labeled comments. For each test comment, I want
to retrieve the most relevant examples dynamically rather than using a fixed set.

Task:
1. Design the retrieval architecture: embedding model, similarity metric,
   and optimal K (number of examples to retrieve).
2. Explain why RAFS outperforms static few-shot for a 41-combination label
   space.
3. Specify how to handle rare labels (severe_toxic, threat, identity_hate)
   that may be underrepresented in top-K nearest neighbours.
4. Write the Python function signature and key logic for the retrieval step.
```

**Response summary:** Recommended `text-embedding-3-small` over TF-IDF, cosine similarity, K=6–7 as optimal, and a hard-label slot guarantee to prevent rare labels being excluded from retrieved examples. Explained that static few-shot cannot cover all 41 combinations while RAFS naturally reflects each query's semantic neighbourhood.

**Decision:** Architecture adopted for Method 3. Initial implementation used TF-IDF embeddings, after testing on 500 rows, the team switched to `text-embedding-3-small` because TF-IDF could not capture semantic severity. This switch was the foundation of the final Method 3 design.

---

#### P-04 · Claude · RAG index construction and `severe_toxic` retrieval fix

**Purpose:** Build the embedding index for the pool and diagnose the `severe_toxic` recall failure in early RAG results.

**Prompt:**

```
You are an expert in RAG systems and NLP classification pipelines.

Context:
- Few-shot pool: few_shot_pool.csv (~151,000 comments, 6 binary labels)
- Test set: stress_test_eval_set.csv (8,000 rows)
- Current approach: centroid selection per label combination for few-shot
  examples
- Current result on 100 samples: severe_toxic F1 = 0.250, Macro F1 = 0.781

Problem: severe_toxic F1 is critically low. Centroid selection picks the most
average severe_toxic example, which is borderline and gives a weak signal.

Task:
1. Diagnose whether the retrieval failure is in Step 1 (index construction)
   or Step 2 (prompt injection), with reasoning.
2. Implement extremal selection for hard labels: pick the comment furthest
   from the clean centroid rather than closest to the label centroid.
3. Add a hard-label slot guarantee: ensure at least 3 of the top-7 retrieved
   examples are positive for severe_toxic, threat, or identity_hate.
4. Write the updated build_index() and retrieve_examples() functions.
```

**Response summary:** Diagnosed Step 1 as the primary failure (centroid method averaged away the severity signal). Implemented hybrid selection, extremal for hard labels, centroid for soft labels, and added the hard-label slot guarantee. Updated retrieval raised `severe_toxic` F1 from 0.250 to 0.662 on n=500.

**Decision:** Extremal selection and hard-label guarantee adopted and carried into Method 3. A cascade post-processing rule that also emerged from this session (`toxic=1 AND obscene=1 AND insult=1 → severe_toxic=1`) was later removed in Method 4 as it conflated model output with rule-based override; it was replaced by the dedicated second-pass severity check prompt (P-07).

---

### Phase 3: Prompt Refinement for Hard Labels

---

#### P-05 · Claude · Diagnosing and fixing `severe_toxic` precision collapse

**Purpose:** Diagnose the `severe_toxic` precision failure (163 FPs on 500 samples) and produce a structured fix.

**Prompt:**

```
You are an expert in multi-label NLP classification and prompt engineering.

Current result on n=500: Macro F1=0.810, severe_toxic precision=0.512 (163 FP).

The prompt currently contains two contradictory instructions:
  "Be conservative for rare labels"
  "Lean toward 1 when uncertain for severe_toxic"

Task:
1. Identify the root cause of the 163 false positives on severe_toxic.
2. Replace the contradictory instructions with a single unambiguous severity
   rubric.
3. Design a 4-point severity scale with concrete score anchors (0–100) and
   at least two "does NOT qualify" examples per anchor.
4. Specify whether a second-pass verification call is warranted for borderline
   cases and define the trigger condition precisely.
```

**Response summary:** Identified the contradictory instructions as the primary cause of FP inflation. Produced the 4-point severity rubric (Mild 10–30 / Moderate 30–50 / Severe 50–80 / Extreme 80–100) with explicit "does NOT qualify" rules. Recommended a second-pass severity check triggered when `toxic ≥ 40 AND 20 ≤ severe_toxic ≤ 65`.

**Decision:** 4-point rubric adopted in the Method 4 main prompt. Second-pass severity check implemented as specified (full prompt in `prompt_templates.md`, Section 5).

---

#### P-06 · ChatGPT · `threat` boundary examples and `identity_hate` coded language

**Purpose:** Reduce `threat` false positives and improve `identity_hate` recall on coded language.

**Prompt:**

```
You are an expert prompt engineer specialising in content moderation
classification.

Two persistent label errors in our GPT-4.1 classifier:

1. threat over-prediction: the model flags rhetorical anger ("I could kill
   for a pizza", "fuck off and die") as genuine threats. Precision = 0.566.

2. identity_hate under-detection: the model misses coded language ("these
   people don't belong here", "go back to your country"). Recall near zero.

Task:
1. Write IS / IS NOT boundary examples for threat that draw a clear line
   between rhetorical anger, hyperbole, and actionable threat. Include the
   requirement for (a) an identifiable target AND (b) a stated harmful action.
2. Write IS / IS NOT boundary examples for identity_hate that cover:
   - Explicit slurs
   - Stereotyping and generalisation
   - Coded/indirect targeting language
   - Critique of a belief vs. attack on a group's members
3. Add COMMON MISTAKE warnings for each label addressing the specific failure
   modes above.
Format all output as prompt-ready text blocks, not prose.
```

**Response summary:** Produced IS/IS NOT blocks and COMMON MISTAKE warnings for both labels (full text incorporated into Method 4 main prompt, `prompt_templates.md`, Section 4). The threat definition introduced the dual requirement of an identifiable target and a stated harmful action.

**Decision:** Both blocks incorporated directly into the Method 4 main prompt. Threat FPs fell by approximately 75% relative to Methods 1 and 3. `identity_hate` recall on coded language improved.

---

#### P-07 · Claude · `identity_hate` second-pass prompt and group-noun detection

**Purpose:** Design the dedicated second-pass identity check for borderline cases and the group-noun heuristic for retrieval boosting.

**Prompt:**

```
You are an expert in multi-label toxicity classification.

identity_hate borderline cases are being missed: the model assigns scores of
15–55 to comments containing coded group-targeting language, falling below
the detection threshold.

Task:
1. Write a focused second-pass classification prompt that asks specifically
   whether a comment attacks people BECAUSE OF their group membership. The
   prompt should:
   - List the protected characteristics explicitly
   - Include probes for coded language patterns
   - Return a single integer 0–100 in JSON format
   - Specify the trigger condition for when this prompt is called

2. Write a Python function contains_group_noun(text) that detects the
   presence of group-identity nouns (ethnicity, religion, gender, sexuality
   terms) to boost the identity_hate retrieval hint score before the main
   classification call.

3. Specify the merge formula for combining the second-pass score with the
   original.
```

**Response summary:** Produced the identity check prompt with trigger condition `hint ≥ 0.08 AND 15 ≤ identity_hate_score ≤ 55`, the `contains_group_noun()` function with a curated noun list, and the 0.5/0.5 merge formula (full prompt in `prompt_templates.md`, Section 4).

**Decision:** Both components adopted in Method 4. `contains_group_noun()` is implemented in the notebook CONFIG section.

---

### Phase 4: Infrastructure and Async Pipeline

---

#### P-08 · Claude · Async inference engine with checkpoint/resume

**Purpose:** Build the production inference engine capable of processing 8,000 rows reliably with rate-limit handling and mid-run recovery.

**Prompt:**

```
You are an expert Python engineer specialising in async API pipelines.

I need to classify 8,000 comments using the OpenAI GPT-4.1 API.
Requirements:
- Async concurrency: 8 simultaneous requests using asyncio.Semaphore
- Rate-limit handling: exponential backoff on 429 and 400 errors,
  up to 5 retries with base delay 1.0s
- Checkpoint/resume: save progress every 500 rows to a CSV so that
  if the run is interrupted, re-running skips completed rows
- Fallback: if the primary prompt fails to parse, send a simplified
  emergency prompt and log the failure to error_log.csv
- Token tracking: record gpt_input_tokens, gpt_output_tokens,
  embed_tokens, and row_runtime_sec for every row

Write the full run_inference() async function and the call_with_backoff()
retry wrapper. Use temperature=0 and seed=42 on every call.
```

**Response summary:** Produced `run_inference()`, `call_with_backoff()`, `process_row()`, and the checkpoint/resume logic. These functions are shared across Methods 1–3 with minor method-specific differences in the prompt builder argument.

**Decision:** Adopted as the shared inference engine across all four methods. The fallback emergency prompt and `error_log.csv` mechanism are visible in the executed output cells of all four notebooks.

---

#### P-09 · Claude · F-string and Jupyter environment debugging

**Purpose:** Resolve two recurring infrastructure errors that blocked notebook execution.

**Prompt:**

```
You are an expert Python and Jupyter notebook engineer.

Two errors are blocking execution:

Error 1:
  SyntaxError: unterminated f-string literal (detected at line 72)
  The problematic line is:
    blocks.append(f'Comment: "{comments[0]}"
    Labels: {json.dumps(labels)}')

Error 2:
  SyntaxError: invalid syntax
  Triggered by running: pip install openai tiktoken
  and: export OPENAI_API_KEY="sk-..."
  directly in Jupyter code cells.

For each error:
1. Explain the root cause.
2. Provide the corrected code.
3. State the general rule to prevent this class of error in future.
```

**Response summary:** Error 1, multi-line f-string with embedded `\n` corrupted during editing; fixed with explicit string concatenation. Error 2, shell commands require `!` prefix in Jupyter cells. Produced corrected code for both.

**Decision:** String concatenation pattern adopted throughout the codebase for multi-line prompt blocks. `!pip install` prefix convention applied in all setup cells.

---

#### P-10 · ChatGPT · Standardised evaluation report formatter

**Purpose:** Produce a consistent evaluation report format usable across all four methods.

**Prompt:**

```
You are an expert Python engineer.

I need a single evaluate_predictions() function that can be called identically
in all four classification notebooks (zero-shot, few-shot, RAG dynamic,
label-aware RAG) and produces a standardised printed report.

The report must include:
- Header: method name, GPT model, timestamp, row count
- Global metrics: Micro and Macro Precision/Recall/F1, Exact Match,
  Hamming Loss
- Per-label table: Label | Prec | Rec | F1 | TP | FP | FN | TN | Support
- ROC-AUC (macro) computed from raw confidence scores, not binary predictions

Format the report with fixed-width alignment so it renders cleanly in
notebook output cells. Use only scikit-learn and numpy.
```

**Response summary:** Produced the `evaluate_predictions()` function with the standardised report format, header block, GLOBAL METRICS section, PER-LABEL METRICS table. Function is identical across all four notebooks.

**Decision:** Adopted as the shared evaluation function. Identical report format enables direct comparison of all four methods from notebook output cells.

---

### Phase 5: Threshold Tuning and Dev/Test Split

---

#### P-11 · Claude · Dev/test split and threshold tuning function

**Purpose:** Implement the stratified dev/test split for Method 4 and the threshold tuning function with label-specific recall floors.

**Prompt:**

```
You are an expert in multi-label classification evaluation.

Method 4 requires:
1. A stratified 50/50 dev/test split of stress_test_eval_set.csv (8,000 rows)
   that preserves label combination proportions. Rare combinations with only
   one example must be forced into the test set.

2. A tune_thresholds_on_dev() function that:
   - Sweeps each label's threshold from 2 to 95 in steps of 2
   - Maximises F1 for each label independently
   - Enforces recall floors: severe_toxic ≥ 0.50, threat ≥ 0.50,
     identity_hate ≥ 0.60
   - Falls back to unconstrained F1 optimisation if the recall floor
     is unachievable at any threshold
   - Saves results to threshold_from_dev_v5_cl1.json

3. A load_dev_thresholds() function that reads the saved JSON and applies
   thresholds to the held-out test set predictions.

Write all three functions. Use sklearn.model_selection.train_test_split
with stratify on the label combination column.
```

**Response summary:** Produced `split_eval_set()`, `tune_thresholds_on_dev()`, and `load_dev_thresholds()`. Dev-set tuning result: `severe_toxic` threshold 75 → 20 (recall 0.808), `threat` threshold 45 → 16 (recall 0.616), `identity_hate` threshold 35 → 2 (recall 0.798).

**Decision:** All three functions adopted in Method 4. The `severe_toxic` threshold drop from 75 to 20 was the single largest driver of Method 4's recall improvement on that label (0.529 → 0.702).

---

#### P-12 · Claude · Skip-gate for high-confidence clean comments

**Purpose:** Implement a cost-saving mechanism that bypasses GPT inference for comments highly likely to be clean.

**Prompt:**

```
You are an expert in LLM inference pipeline optimisation.

In Method 4, approximately 50% of the evaluation set is clean (no toxic
labels). For many of these, the top-8 nearest neighbours in the pool are all
clean comments.

Design a skip-gate that bypasses the GPT classification call when:
- All 8 nearest pool neighbours are clean (no positive labels)
- The comment contains no toxic keywords from a defined keyword list
- The comment has more than 12 words (short comments are ambiguous)

If all three conditions are met, assign all-zero predictions directly
without calling the API.

Requirements:
1. Write the is_skip_eligible(comment, neighbours, hints) function.
2. Define the SKIP_CLEAN_RATIO threshold (proportion of clean neighbours
   required).
3. Ensure the skip decision is logged per row so bypass rate can be audited.
4. Confirm the skip-gate does not affect rows where any hard-label hint
   > 0.05.
```

**Response summary:** Produced `is_skip_eligible()` with `SKIP_CLEAN_RATIO=0.95`, the toxic keyword list, the word-count floor of 12, and per-row logging. Hard-label hint protection (`HARD_LABEL_HINT_FLOOR=0.08`) confirmed.

**Decision:** Skip-gate adopted. On the 4,001-row test split, the gate bypassed 69 comments (1.7%), saving approximately $0.13 in API cost without affecting classification quality on toxic comments.

---
## 5. Final Results Summary

All four methods were evaluated on the same 8,000-row stress-test evaluation set (50/50 toxic/clean split, rare-class over-sampled). Method 4 used a held-out 4,001-row test split with thresholds tuned on the 3,999-row dev split.

| Method | Micro F1 | Macro F1 | Exact Match | ROC-AUC (macro) | Key Change |
|---|---|---|---|---|---|
| M1: Zero-Shot | 0.821 | 0.773 | 0.587 | 0.874 | Baseline |
| M2: Few-Shot Static | 0.786 | 0.723 | 0.575 | 0.834 | −5.0pp (context dilution) |
| M3: RAG Dynamic | 0.827 | 0.775 | 0.599 | 0.918 | +5.2pp vs M2 |
| M4: Label-Aware RAG | 0.829 | 0.774 | 0.613 | — | Best exact match (4k test split) |

---

*Document prepared by the group. All prompts in Section 4 are structured reconstructions of substantive exchanges from 15+ conversations held between 4 February and 16 March 2026.*
