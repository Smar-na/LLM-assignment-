# Vibe Coding Prompt Log

**Project:** LLM-Based Toxic Comment Classification

## Table of Contents

1. [Statement of AI Use](#1-statement-of-ai-use)
2. [AI Tools Used](#2-ai-tools-used)
3. [Prompt Iteration Log: Cross-Reference](#3-prompt-iteration-log-how-the-classification-prompts-evolved)
4. [Structured Prompt Log](#4-structured-prompt-log)
5. [BERT Baseline Prompts](#5-bert-baseline-prompts)
6. [Final Results Summary](#6-final-results-summary)

## 1. Statement of AI Use

AI assistance was used across more than 15 individual conversations spread across team members, covering all four GPT-4.1 pipeline methods and the BERT baseline. Rather than submitting raw chat transcripts, which span thousands of lines across multiple sessions, this document presents each substantive prompt as a structured entry in Section 4, recording the purpose, full prompt text, model response summary, and the decision that followed. Section 3 cross-references `prompt_templates_improved.md` Section 11, which is the authoritative record of what changed between methods and why. All sessions were conducted between 4 February and 16 March 2026.

## 2. AI Tools Used

| Tool | Version | Primary Use |
|---|---|---|
| ChatGPT | GPT-4.1 (OpenAI) | All four classification inference pipelines (M1 to M4); few-shot example design; RAG architecture discussions; prompt iteration for `severe_toxic` and `threat` labels; evaluation format specification |
| Claude | Sonnet (Anthropic) | Coding assistance throughout; RAG index construction debugging; f-string and async pipeline fixes; threshold tuning function development; BERT inference pipeline setup |

> **Correction from earlier draft:** A previous version of this table incorrectly listed Claude Sonnet as the inference engine for Methods 1, 3, and 4. All four GPT-4.1 classification methods use the OpenAI API (`gpt-4.1`, `temperature=0`, `seed=42`) as confirmed by the `GPT_MODEL = "gpt-4.1"` configuration in each notebook. Claude was used exclusively for coding assistance, debugging, and function development.

## 3. Prompt Iteration Log: How the Classification Prompts Evolved

The full iteration log, covering the design decision made at each stage (M1 to M4), the problem that motivated it, and its measured effect on performance, is documented in `prompt_templates_improved.md`, Section 11 (Design Rationale and Iteration Log), which is the authoritative reference for this material. The development decisions that drove each iteration are also traceable through the Purpose and Decision fields of the structured prompt entries in Section 4 of this document (P-01 through P-15).

## 4. Structured Prompt Log

This section documents every substantive prompt used during development as a structured engineering artefact. Each entry records the prompt's **purpose**, the **full prompt text** as sent to the model, the **model response summary**, and the **decision or outcome** that followed. Ad-hoc debugging turns (syntax errors, pip install issues, file path corrections) are excluded; only prompts that produced a design decision or a code component that persists in the final submission are included.

Prompts are grouped by development phase and numbered sequentially (P-01 through P-15) for cross-referencing.

### Phase 1: Project Framing and Output Format Design

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

**Decision:** The team identified that the single-line label format would make threshold tuning and ROC-AUC computation impossible. Output format was changed to JSON confidence scores (0 to 100) for all labels in Methods 2 to 4, enabling per-label threshold calibration and ROC-AUC reporting. Method 1 retained binary 0/1 output with a post-processing rescale step. This pivot drove the architecture of all four methods.

#### P-02 · ChatGPT · Zero-shot prompt component design

**Purpose:** Define the specific components that make a zero-shot prompt effective for a 41-combination multi-label dataset, and produce a structured template.

**Prompt:**

```
You are an expert prompt engineer.

I am building a zero-shot prompt for GPT-4.1 to perform multi-label toxicity
classification on the Jigsaw dataset. The six labels are:
toxic, severe_toxic, obscene, threat, insult, identity_hate.

The dataset has 41 unique label combinations. The model must output a JSON
object with a confidence score (0-100) for each label.

Task:
1. List the essential components of an effective zero-shot prompt for this
   task, with a justification for each component.
2. Identify the three most common label confusion errors in toxicity
   classification and write decision guidelines that address each.
3. Produce a complete, production-ready zero-shot prompt template.
   Use temperature=0 and seed=42 for reproducibility.
```

**Response summary:** Identified five components: task definition, label schema with decision boundaries, output format constraint, decision rules, and deterministic settings. Addressed `insult` vs `identity_hate` confusion and flagged `threat` over-prediction from angry language. Produced the structured template that became Method 1 (full template in `prompt_templates_improved.md`, Section 1).

**Decision:** Template adopted as the Method 1 baseline. Decision guidelines were carried forward into Methods 2 to 4 with progressive refinement. Note that Method 1 requests binary 0/1 output rather than confidence scores, which is a design choice explained in `prompt_templates_improved.md` Section 11.4.

### Phase 2: RAG Architecture and Retrieval Design

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

**Response summary:** Recommended `text-embedding-3-small` over TF-IDF and cosine similarity. Suggested K=6 to 7 as the optimal range and a hard-label slot guarantee to prevent rare labels being excluded from retrieved examples. Explained that static few-shot cannot cover all 41 combinations while RAFS naturally reflects each query's semantic neighbourhood.

**Decision:** Architecture adopted for Method 3 (`METHOD3_RAGDYNAMIC.ipynb`). The final implementation uses `RAG_K = 7` total examples with `RAG_HARD_K = 3` slots guaranteed for hard-minority-label positives (`severe_toxic`, `threat`, `identity_hate`). Initial implementation used TF-IDF embeddings; after testing on 500 rows, the team switched to `text-embedding-3-small` because TF-IDF could not capture semantic severity. This switch was the foundation of the final Method 3 design. Method 4 (`METHOD_4_RAG_HybridAugmentation.ipynb`) uses `RETRIEVE_K = 6` as default and `RETRIEVE_K_HIGH = 8` when hard-label hints are detected.

#### P-04 · Claude · RAG index construction and `severe_toxic` retrieval fix

**Purpose:** Build the embedding index for the pool and diagnose the `severe_toxic` recall failure in early RAG results.

**Prompt:**

```
You are an expert in RAG systems and NLP classification pipelines.

Context:
- Few-shot pool: few_shot_pool_fixed_3.csv (~151,000 comments, 6 binary labels)
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

**Response summary:** Diagnosed Step 1 as the primary failure (centroid method averaged away the severity signal). Implemented hybrid selection, using extremal for hard labels and centroid for soft labels, and added the hard-label slot guarantee. Updated retrieval raised `severe_toxic` F1 from 0.250 to 0.662 on n=500.

**Decision:** Extremal selection and hard-label guarantee adopted and carried into Method 3. A cascade post-processing rule that also emerged from this session (`toxic=1 AND obscene=1 AND insult=1` implies `severe_toxic=1`) was later removed in Method 4 as it conflated model output with rule-based override; it was replaced by the dedicated second-pass severity check prompt (P-07).

### Phase 3: Prompt Refinement for Hard Labels

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
3. Design a 4-point severity scale with concrete score anchors (0-100) and
   at least two "does NOT qualify" examples per anchor.
4. Specify whether a second-pass verification call is warranted for borderline
   cases and define the trigger condition precisely.
```

**Response summary:** Identified the contradictory instructions as the primary cause of FP inflation. Produced the 4-point severity rubric (Mild 10 to 30 / Moderate 30 to 50 / Severe 50 to 80 / Extreme 80 to 100) with explicit "does NOT qualify" rules. Recommended a second-pass severity check triggered when `toxic >= 40 AND 20 <= severe_toxic <= 65`.

**Decision:** 4-point rubric adopted in the Method 4 main prompt. Second-pass severity check implemented as specified (full prompt in `prompt_templates_improved.md`, Section 5).

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

**Response summary:** Produced IS/IS NOT blocks and COMMON MISTAKE warnings for both labels (full text incorporated into Method 4 main prompt, `prompt_templates_improved.md`, Section 4). The threat definition introduced the dual requirement of an identifiable target and a stated harmful action.

**Decision:** Both blocks incorporated directly into the Method 4 main prompt. Based on the per-label results in Table 3.6 of the report, `threat` false positives on the 4,001-row test split were 74 (M4), compared to 300 on the full 8,000-row set for M1 and 314 for M3. Adjusting for the different row counts, the effective FP rate reduction is approximately 50%, not the 75% figure cited in earlier drafts of this document. `identity_hate` recall on coded language improved, as noted in the report's cross-method comparison (Section 3.3.2).

#### P-07 · Claude · `identity_hate` second-pass prompt and group-noun detection

**Purpose:** Design the dedicated second-pass identity check for borderline cases and the group-noun heuristic for retrieval boosting.

**Prompt:**

```
You are an expert in multi-label toxicity classification.

identity_hate borderline cases are being missed: the model assigns scores of
15-55 to comments containing coded group-targeting language, falling below
the detection threshold.

Task:
1. Write a focused second-pass classification prompt that asks specifically
   whether a comment attacks people BECAUSE OF their group membership. The
   prompt should:
   - List the protected characteristics explicitly
   - Include probes for coded language patterns
   - Return a single integer 0-100 in JSON format
   - Specify the trigger condition for when this prompt is called

2. Write a Python function contains_group_noun(text) that detects the
   presence of group-identity nouns (ethnicity, religion, gender, sexuality
   terms) to boost the identity_hate retrieval hint score before the main
   classification call.

3. Specify the merge formula for combining the second-pass score with the
   original.
```

**Response summary:** Produced the identity check prompt with trigger condition `hint >= 0.08 AND 15 <= identity_hate_score <= 55`, the `contains_group_noun()` function with a curated noun list (see `prompt_templates_improved.md` Appendix B), and the 0.5/0.5 merge formula (full prompt in `prompt_templates_improved.md`, Section 6).

**Decision:** Both components adopted in Method 4. The group-noun list and hint-boosting logic are implemented in the notebook CONFIG section of `METHOD_4_RAG_HybridAugmentation.ipynb`.

### Phase 4: Infrastructure and Async Pipeline

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

**Response summary:** Produced `run_inference()`, `call_with_backoff()`, `process_row()`, and the checkpoint/resume logic. These functions are shared across Methods 1 to 3 with minor method-specific differences in the prompt builder argument.

**Decision:** Adopted as the shared inference engine across all four methods. The fallback emergency prompt and `error_log.csv` mechanism are visible in the executed output cells of all four notebooks.

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

**Response summary:** Error 1 was a multi-line f-string with embedded `\n` that was corrupted during editing; fixed with explicit string concatenation. Error 2 was caused by running shell commands without the `!` prefix in Jupyter cells. Produced corrected code for both.

**Decision:** String concatenation pattern adopted throughout the codebase for multi-line prompt blocks. `!pip install` prefix convention applied in all setup cells.

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

**Response summary:** Produced the `evaluate_predictions()` function with the standardised report format, header block, GLOBAL METRICS section, and PER-LABEL METRICS table. The function is shared across Methods 1 to 3. Note that Method 4 uses a separate dev/test split and does not call this function in the same way as Methods 1 to 3, which is why ROC-AUC is reported as not available for M4 in the results tables.

**Decision:** Adopted as the shared evaluation function for Methods 1 to 3. The identical report format enables direct comparison of those three methods from notebook output cells.

### Phase 5: Threshold Tuning and Dev/Test Split

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
   - Enforces recall floors: severe_toxic >= 0.50, threat >= 0.50,
     identity_hate >= 0.60
   - Falls back to unconstrained F1 optimisation if the recall floor
     is unachievable at any threshold
   - Saves results to threshold_from_dev_v5_cl1.json

3. A load_dev_thresholds() function that reads the saved JSON and applies
   thresholds to the held-out test set predictions.

Write all three functions. Use sklearn.model_selection.train_test_split
with stratify on the label combination column.
```

**Response summary:** Produced `split_eval_set()`, `tune_thresholds_on_dev()`, and `load_dev_thresholds()`. Dev-set tuning output: `severe_toxic` threshold 35 to 20 (recall 0.808 on dev), `threat` threshold 45 to 16 (recall 0.616 on dev), `identity_hate` threshold 35 to 2 (recall 0.798 on dev). Results saved to `threshold_from_dev_v5_cl1.json`.

**Decision:** All three functions adopted in Method 4. The `severe_toxic` threshold drop from 35 (initial) to 20 (dev-tuned), which compares to the M1 to M3 fixed threshold of 75, was the single largest driver of Method 4's recall improvement on that label (0.529 to 0.702), as reported in Section 3.2.4 of the report.

#### P-12 · Claude · Skip-gate for high-confidence clean comments

**Purpose:** Implement a cost-saving mechanism that bypasses GPT inference for comments highly likely to be clean.

**Prompt:**

```
You are an expert in LLM inference pipeline optimisation.

In Method 4, approximately 50% of the evaluation set is clean (no toxic
labels). For many of these, the top-K nearest neighbours in the pool are all
clean comments.

Design a skip-gate that bypasses the GPT classification call when:
- The nearest pool neighbours are predominantly clean (no positive labels)
- The comment contains no toxic keywords from a defined keyword list
- The comment has more than 12 words (short comments are ambiguous)

If all conditions are met, assign all-zero predictions directly
without calling the API.

Requirements:
1. Write the is_skip_eligible(comment, neighbours, hints) function.
2. Define the SKIP_CLEAN_RATIO threshold (proportion of clean neighbours
   required).
3. Ensure the skip decision is logged per row so bypass rate can be audited.
4. Confirm the skip-gate does not affect rows where any hard-label hint
   > 0.05.
```

**Response summary:** Produced `is_skip_eligible()` with `SKIP_CLEAN_RATIO = 0.95`, the toxic keyword list, the word-count floor of 12 words (`SHORT_COMMENT_WORDS = 12`), and per-row logging. Hard-label hint protection (`HARD_LABEL_HINT_FLOOR = 0.08`) confirmed. The skip-gate operates on the default retrieval pool of 6 neighbours (`RETRIEVE_K = 6`), as the 8-neighbour expanded pool (`RETRIEVE_K_HIGH = 8`) is only used when hard-label hints are detected, which by definition cannot be the case for skip-eligible comments.

**Decision:** Skip-gate adopted. On the 4,001-row test split, the gate bypassed 69 comments (1.7%), saving approximately $0.13 in API cost without affecting classification quality on toxic comments.

## 5. BERT Baseline Prompts

The BERT notebook (`BERT.ipynb`) was developed to provide supervised baselines for comparison with the GPT-4.1 pipeline. All BERT inference runs on the 8,000-row stress-test evaluation set (`stress_test_eval_set.csv`), the same set used by Methods 1 to 3.

### Phase 6: BERT Baseline Design and Inference

#### P-13 · Claude · BERT model selection and classifier architecture

**Purpose:** Establish which BERT variant to use, justify the architecture choices, and implement the inference pipeline.

**Prompt:**

```
You are an expert in NLP classification using transformer models.

We are adding a BERT-based baseline to compare against our GPT-4.1
classification pipeline on the Jigsaw toxicity dataset.
The six labels are: toxic, severe_toxic, obscene, threat, insult, identity_hate.
The evaluation set is stress_test_eval_set.csv (8,000 rows).

Task:
1. Recommend which BERT variant to use, with justification:
   - google-bert/bert-base-cased (general pre-trained, no toxicity fine-tuning)
   - unitary/toxic-bert (fine-tuned directly on the Jigsaw dataset)
2. Explain why sigmoid activation is appropriate for multi-label classification
   rather than softmax, and why top_k=None is needed in the HuggingFace pipeline.
3. Write the setup_classifier() function using AutoTokenizer and
   AutoModelForSequenceClassification with num_labels=6.
4. Write the run_inference() function using the HuggingFace pipeline
   with batch_size=32, truncation=True, and max_length=512.
5. Explain why bert-base-cased with a randomly initialised classification head
   is only valid as a lower-bound baseline, and what the load warning
   "classifier.weight MISSING" means in practice.
```

**Response summary:** Recommended `unitary/toxic-bert` as the primary BERT comparison because it was fine-tuned directly on the Jigsaw dataset, giving it domain-specific supervision. Confirmed `google-bert/bert-base-cased` can be included as a random-weight lower-bound baseline, with the explicit caveat that its `classifier.weight MISSING` load warning confirms the head is randomly initialised and its outputs are not meaningful toxicity predictions. Explained that `sigmoid` is appropriate for multi-label tasks because labels are independent, and `top_k=None` returns all six label scores rather than only the highest. Produced `setup_classifier()` and `run_inference()` using the HuggingFace pipeline.

**Decision:** Both models retained in `BERT.ipynb`, with `bert-base-cased` explicitly labelled as a lower-bound baseline in the report (Section 3.1.1) to isolate the contribution of general linguistic pre-training from task-specific supervision. `unitary/toxic-bert` is treated as the primary supervised BERT benchmark. A fixed threshold of 0.5 is applied uniformly across all six labels for both models, with no dev-set threshold tuning, as BERT is used for comparison rather than optimisation.

#### P-14 · Claude · BERT evaluation and metric alignment

**Purpose:** Adapt the evaluation output for BERT and align it with the GPT-method reporting format.

**Prompt:**

```
You are an expert in multi-label classification evaluation.

The BERT HuggingFace pipeline returns scores as:
[{'label': 'LABEL_0', 'score': 0.73}, {'label': 'LABEL_1', 'score': 0.12}, ...]

For unitary/toxic-bert, the labels resolve as:
{0: 'toxic', 1: 'severe_toxic', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_hate'}

For google-bert/bert-base-cased with num_labels=6, the labels resolve as:
{0: 'LABEL_0', 1: 'LABEL_1', ..., 5: 'LABEL_5'}

Task:
1. Write the code to sort scores by label name and map them to the correct
   label order: toxic, severe_toxic, obscene, threat, insult, identity_hate.
2. Apply a fixed threshold of 0.5 to produce binary predictions.
3. Generate a per-label classification_report using sklearn with
   zero_division=0 and output_dict=True.
4. Explain why a fixed 0.5 threshold is used for BERT while GPT methods
   used per-label tuned thresholds, and what effect this has on comparability.
5. Confirm which metrics cannot be computed for BERT given this setup
   (specifically Hamming Loss and ROC-AUC), and why.
```

**Response summary:** Produced the score-sorting and label-mapping logic (implemented via `sorted(result, key=lambda x: x['label'])` in `run_inference()`). Confirmed that a fixed 0.5 threshold is appropriate for a baseline comparison but reduces BERT's performance relative to what threshold tuning could achieve, meaning BERT numbers represent a conservative lower bound. Confirmed that Hamming Loss is not computed in `BERT.ipynb` as the notebook uses `classification_report` from sklearn rather than the shared `evaluate_predictions()` function used in Methods 1 to 3, meaning the Hamming Loss values for BERT that appear in Table 3.8 of the report are estimates and should be treated accordingly.

**Decision:** BERT evaluation uses `classification_report` with `zero_division=0`, as implemented in Cells 9 and 15 of `BERT.ipynb`. All BERT metrics reported in Tables 3.1 and 3.8 of the report are taken from the stress-test set outputs in those cells and verified against the notebook output. The Hamming Loss figures for BERT in Table 3.8 were computed separately and are noted as estimates.

#### P-15 · Claude · BERT timing and cost comparison

**Purpose:** Establish the inference time data for BERT to populate Table 3.9 of the report and contextualise the cost comparison with GPT methods.

**Prompt:**

```
You are an expert in production ML inference benchmarking.

We are comparing BERT and GPT-4.1 inference on the same 8,000-row dataset.

BERT setup:
- google-bert/bert-base-cased and unitary/toxic-bert
- HuggingFace pipeline, batch_size=32, GPU (Colab T4)
- Measured inference time: approximately 205 seconds per model on 8,000 rows

GPT-4.1 setup:
- OpenAI API, async concurrency=8, temperature=0, seed=42
- M1: ~21 minutes, M2: ~18 minutes, M3: ~22 minutes, M4: ~18.5 minutes

Task:
1. Write the timing instrumentation using time.perf_counter() to measure
   total wall time, setup time, and inference time separately.
2. Compute throughput (samples/sec) and average latency (ms/sample).
3. Explain the key operational difference between BERT (local GPU, no API cost)
   and GPT (API-dependent, per-token cost) for deployment purposes.
```

**Response summary:** Produced the timing wrapper using `time.perf_counter()` (implemented in Cells 8 and 14 of `BERT.ipynb`). Confirmed that BERT throughput on the Colab T4 GPU is approximately 38 to 39 samples per second with average latency around 25 to 26 ms per sample. Explained that BERT runs at zero API cost with predictable latency, while GPT methods incur per-token costs and depend on API availability and rate limits.

**Decision:** Timing instrumentation adopted as shown in `BERT.ipynb`. The inference times in Table 3.9 of the report reflect total wall time including model setup and download, which may differ slightly from the inference-only figures shown in the notebook output cells.

## 6. Final Results Summary

All four GPT-4.1 methods were evaluated on the 8,000-row stress-test evaluation set (50/50 toxic/clean split, rare-class over-sampled). Method 4 used a stratified dev/test split of the same set: 3,999 rows for threshold tuning and 4,001 held-out rows for evaluation. BERT baselines were evaluated on the same 8,000-row stress-test set using a fixed threshold of 0.5. Direct comparison of BERT and GPT-4.1 figures should be interpreted with caution given the different threshold strategies.

| Method | Micro F1 | Macro F1 | Exact Match | ROC-AUC (Micro) | Key Change |
|---|---|---|---|---|---|
| BERT-Base Cased | 0.295 | 0.231 | n/a | n/a | Random-weight lower bound |
| Toxic-BERT | 0.490 | 0.434 | n/a | n/a | Fine-tuned supervised baseline |
| M1: Zero-Shot | 0.821 | 0.773 | 0.587 | 0.895 | GPT-4.1 baseline |
| M2: Few-Shot Static | 0.786 | 0.723 | 0.575 | 0.844 | 5.0pp drop (context dilution) |
| M3: RAG Dynamic | 0.827 | 0.775 | 0.599 | 0.930 | Best ROC-AUC; 5.2pp gain vs M2 |
| M4: HA-RAG | 0.829 | 0.774 | 0.613 | n/a | Best exact match (4,001-row test split) |

> **Note on ROC-AUC values:** Earlier drafts of this document reported M1 ROC-AUC as 0.874, M2 as 0.834, and M3 as 0.918. These were incorrect. The correct values are taken from the executed output cells of each notebook and match Table 3.2 of the report. ROC-AUC is not available for Method 4 because the dev/test split changes the score distribution, making direct comparison with the M1 to M3 full-set figures unreliable.

*Document prepared by the group. All prompts in Section 4 are structured reconstructions of substantive exchanges from 15+ conversations held between 4 February and 16 March 2026.*
