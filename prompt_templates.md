# Prompt Templates: Toxic Comment Classification

This document contains all prompt templates used across the four GPT-4.1 classification pipelines. Each template is annotated with its design rationale. Templates are referenced in the report (Section 2.3) and implemented in the corresponding notebooks.

**Model:** GPT-4.1 (`temperature=0`, `seed=42`, `max_tokens=150`)  
**Output format:** M1 returns binary 0/1 JSON (rescaled to 0/100 before threshold application); M2 to M4 return JSON confidence scores (0 to 100 scale) converted to binary predictions via per-label thresholds  
**Labels:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

## Table of Contents

1. [Method 1: Zero-Shot Prompt](#1-method-1-zero-shot-prompt)
2. [Method 2: Few-Shot Static Prompt](#2-method-2-few-shot-static-prompt)
3. [Method 3: RAG Dynamic Prompt](#3-method-3-rag-dynamic-prompt)
4. [Method 4: HA-RAG Prompt (Main Pass)](#4-method-4-ha-rag-prompt-main-pass)
5. [Method 4: Severity Check Prompt (Second Pass)](#5-method-4-severity-check-prompt-second-pass)
6. [Method 4: Identity Hate Check Prompt (Second Pass)](#6-method-4-identity-hate-check-prompt-second-pass)
7. [Method 4: Self-Consistency Pass](#7-method-4-self-consistency-pass)
8. [Method 4: Skip-Gate Logic (No Prompt)](#8-method-4-skip-gate-logic-no-prompt)
9. [Method 4: Post-Hoc Calibration Rules](#9-method-4-post-hoc-calibration-rules)
10. [Method 4: Initial vs Dev-Tuned Thresholds](#10-method-4-initial-vs-dev-tuned-thresholds)
11. [Design Rationale and Iteration Log](#11-design-rationale-and-iteration-log)

## 1. Method 1: Zero-Shot Prompt

**File:** `METHOD1_zero_shot_.ipynb`  
**Strategy:** Zero-shot with role assignment, label definitions, decision guidelines, and strict JSON output enforcement  
**System message:** `You are a strict multi-label text classifier specialising in toxic content detection for online comments.`

```
SYSTEM ROLE: You are a strict multi-label text classifier specialising in toxic content detection for online comments.

TASK: Classify the given comment into six toxicity categories using multi-label classification.
Each category must be assigned a binary value:
• 1 = label is present
• 0 = label is not present

IMPORTANT:
• Labels are NOT mutually exclusive.
• A comment can have multiple labels = 1.
• If the comment contains no toxic content, all labels must be 0.
• Do NOT provide explanations.
• Do NOT infer beyond the text.
• Be conservative and evidence-based, especially for rare labels.

LABEL DEFINITIONS:
1. toxic: General rude, disrespectful, or offensive language.
2. severe_toxic: Extremely aggressive, abusive, or highly hostile language.
3. obscene: Profanity, vulgar, or sexually explicit language.
4. threat: Statements that imply violence, harm, or intimidation.
5. insult: Direct personal attacks, humiliation, or degrading remarks toward a person or group.
6. identity_hate: Hate speech targeting protected characteristics (e.g., race, religion, gender, nationality, ethnicity, sexual orientation).

DECISION GUIDELINES (CRITICAL FOR CONSISTENCY):
• Assign "threat" = 1 only if there is a clear indication of harm or violence.
• Assign "identity_hate" = 1 only if hatred targets an identity group, not just an individual.
• Profanity alone usually indicates "obscene" and/or "toxic", not automatically "severe_toxic".
• Strong personal attack should be labelled "insult" (and possibly "toxic").
• Extremely abusive language may include "severe_toxic" in addition to other labels.
• Absence of offensive cues should return all zeros.

OUTPUT FORMAT (STRICT, JSON ONLY):
Return ONLY a valid JSON object with EXACTLY these keys:
{
"toxic": 0,
"severe_toxic": 0,
"obscene": 0,
"threat": 0,
"insult": 0,
"identity_hate": 0
}

NO extra text. NO explanations. NO additional fields.

Comment: "{comment}"
```

**Output format note:** Method 1 requests binary 0/1 labels directly. The `extract_scores()` function detects binary output and rescales it to 0 or 100 before applying thresholds, so thresholds are applied on the rescaled 0 to 100 values. In practice, for binary output, the threshold values below only control which label (0 or 100) triggers a positive prediction, as any threshold below 100 will pass a positive through.

**Thresholds applied (to rescaled 0 to 100 scores):** `toxic=25`, `severe_toxic=75`, `obscene=40`, `threat=40`, `insult=50`, `identity_hate=55`

## 2. Method 2: Few-Shot Static Prompt

**File:** `METHOD2_few_shot_simple.ipynb`  
**Strategy:** Static few-shot with 41 curated real examples covering all label combinations in the Jigsaw dataset  
**System message:** `You are a strict multi-label text classifier specialising in toxic content detection for online comments.`

> **Note:** The `{FEW_SHOT_BLOCK}` placeholder is replaced at runtime with 41 formatted examples drawn from `few_shot_pool_fixed_3.csv`. Each example includes the comment text and its confidence scores (binary labels scaled to 0 or 100). Examples cover all 41 unique label combinations observed in the dataset.

```
You are a strict multi-label text classifier specialising in toxic content detection for online comments.

Labels and definitions:
- toxic: rude, hateful, or aggressive language
- severe_toxic: extreme abuse or dehumanising hostility (STRICTER than toxic; dehumanisation, calls for violence)
- obscene: explicit profanity or vulgar sexual language
- threat: real, specific implication of harm toward a person (NOT general anger or hyperbole)
- insult: direct personal attack on the person addressed
- identity_hate: attack targeting a protected GROUP identity (race, religion, gender, sexuality)

Decision rules:
1. severe_toxic is a strict subset of toxic.
2. threat requires clear intent, not just angry language.
3. identity_hate must target GROUP identity, not just insult an individual.
4. Multi-label output is allowed.

Reference examples (use these to calibrate your confidence scores):
{FEW_SHOT_BLOCK}

Classify this comment:
{comment}

Output ONLY this JSON with confidence scores 0-100 for each label and nothing else:
{"scores": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}}
```

**Thresholds applied:** `toxic=25`, `severe_toxic=75`, `obscene=40`, `threat=40`, `insult=50`, `identity_hate=55`

## 3. Method 3: RAG Dynamic Prompt

**File:** `METHOD3_RAGDYNAMIC.ipynb`  
**Strategy:** Per-query dynamic retrieval via `text-embedding-3-small` cosine similarity; top-7 examples injected per comment, with a minimum of 3 slots reserved for hard-minority-label positives (`severe_toxic`, `threat`, `identity_hate`)  
**System message:** `You are a precise multi-label toxicity classification assistant. Return valid JSON only.`

> **Note:** `{retrieved_block}` is replaced at runtime with the top-7 most semantically similar examples retrieved from the pool for each individual comment. At least 3 of the 7 slots are guaranteed to contain hard-label positives (`severe_toxic`, `threat`, `identity_hate`), implemented via the `RAG_HARD_K = 3` parameter in the notebook config. Each example shows the comment, its similarity score, and its label scores. `{n_retrieved}` is the actual number of examples retrieved (up to 7).

```
You are a precise multi-label toxicity classification assistant. Return valid JSON only.

Your task is to assign a confidence score from 0 to 100 for each toxicity label.

LABELS:
- toxic: rude, aggressive, hostile, or generally offensive language
- severe_toxic: extreme abuse, dehumanising hostility, or very intense toxic language
- obscene: explicit profanity, vulgarity, or sexually explicit/offensive wording
- threat: statement implying physical harm, violence, intimidation, or a clear wish to injure
- insult: direct personal attack, humiliation, ridicule, or degrading statement toward a person
- identity_hate: attack, contempt, or hate directed at a protected group identity
  (e.g. race, religion, ethnicity, nationality, gender, sexual orientation)

IMPORTANT DECISION RULES:
1. Multi-label classification is allowed. More than one label may be present.
2. severe_toxic is a strict subset of toxic.
   - If severe_toxic is high, toxic should also be high.
3. threat requires a genuine implication of harm or violence.
   - Mere anger, profanity, or "I hate you" is not enough.
4. identity_hate requires hostility toward a protected group identity.
   - Attacking one individual without group-based targeting is not identity_hate.
5. Profanity alone may indicate obscene and/or toxic, but not automatically severe_toxic.
6. insult requires a direct degrading or attacking expression toward a person or target.
7. Be conservative for rare labels, especially threat and identity_hate.
8. Base the decision only on the text provided. Do not infer hidden intent beyond the wording.

HOW TO USE THE REFERENCE EXAMPLES:
- The {n_retrieved} examples below were retrieved specifically for this comment
  using semantic similarity (text-embedding-3-small cosine search).
- Similarity scores are shown, where higher means more semantically related.
- Use them to calibrate label thresholds, not to copy scores mechanically.

RETRIEVED EXAMPLES (ordered by similarity):
{retrieved_block}

Now classify this comment:
"""{comment}"""

SCORING GUIDANCE:
- 0-10: label clearly absent
- 11-39: weak or ambiguous evidence
- 40-59: borderline / uncertain presence
- 60-79: reasonably clear presence
- 80-100: very clear presence

OUTPUT REQUIREMENTS:
- Return ONLY one valid JSON object.
- Use EXACTLY the keys shown below.
- Do not include explanations, markdown, or extra text.

{
  "scores": {
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
  }
}
```

**Thresholds applied:** `toxic=25`, `severe_toxic=75`, `obscene=40`, `threat=40`, `insult=50`, `identity_hate=55`

## 4. Method 4: HA-RAG Prompt (Main Pass)

**File:** `METHOD_4_RAG_HybridAugmentation.ipynb`  
**Strategy:** Label-aware retrieval with purity-ranked examples, 4-point severity rubric, boundary examples for each label, and explicit failure-mode warnings. This method is referred to throughout as HA-RAG (Hybrid Augmentation RAG) in the report.  
**System message:** `You are a precise multi-label toxicity classification assistant. Return valid JSON only.`

> **Note:** `{examples_block}` is replaced at runtime with label-aware retrieved examples formatted with label strings and similarity scores. `{comment}` is the target comment. On self-consistency passes, the intro line changes to `"Re-evaluate this comment carefully, paying extra attention to severity and context:"` (see Section 7).

```
You are a precise multi-label toxicity classification assistant. Return valid JSON only..

Labels and definitions (read carefully, common mistakes noted):

- toxic: rude, hateful, or aggressive language that could make someone want to leave a conversation.

- severe_toxic: SIGNIFICANTLY more intense than ordinary toxic. Use this severity scale:
  * Mild toxic (score 10-30): single profanity, casual rudeness, mild name-calling
  * Moderate toxic (score 30-50): repeated profanity, aggressive hostility, personal attacks
  * Severe toxic (score 50-80): sustained extreme abuse, dehumanising language, violent hostility,
    intense degrading attacks, or repeated extreme obscene abuse directed at a person
  * Extreme toxic (score 80-100): explicit calls for harm combined with dehumanisation
  COMMON MISTAKE: Do not require genocide-level language for severe_toxic. Intense, sustained
  personal abuse with extreme profanity IS severe_toxic even without extermination language.
  ALSO: A single profane word alone is NOT severe_toxic — severity requires intensity OR persistence.

- obscene: explicit profanity, vulgar sexual language, or graphic crude language.

- threat: a CREDIBLE, SPECIFIC indication of intent to cause physical harm to a person.
  REQUIRES: (a) an identifiable target AND (b) a stated harmful action.
  IS a threat: "I will find you and kill you", "you're going to get shot"
  IS NOT a threat: "I could kill for a pizza" (hyperbole), "fuck off and die" (anger expression),
  "someone should teach them a lesson" (vague, no specific target or action),
  "I hope you get hit by a bus" (wish, not stated intent to act).
  COMMON MISTAKE: Angry language is NOT a threat unless it contains specific intent to harm.

- insult: a direct personal attack on someone's character, intelligence, or worth.
  Must target a specific person or addressee, not a general statement.

- identity_hate: attacks targeting a person BECAUSE OF their membership in a protected group
  (race, religion, ethnicity, gender, sexual orientation, nationality, disability).
  IS identity_hate: "all Muslims are terrorists", "go back to your country", slurs targeting a group
  IS NOT identity_hate: "you're an idiot" (personal insult, not group-based),
  "I disagree with Islam" (critique of religion, not attack on people)
  COMMON MISTAKE: Missing coded/indirect hate speech like "these people don't belong here"
  when referring to an ethnic or religious group.

Decision rules:
1. Multi-label output is expected, as most toxic comments have 2-4 labels active.
2. If severe_toxic is positive, toxic must also be positive.
3. Score each label independently based on its definition above.
4. Pay attention to the INTENSITY and PERSISTENCE of language, not just the presence of bad words.

Reference examples (study the severity gradients):
{examples_block}

Classify this comment:
{comment}

Output ONLY this JSON with confidence scores 0-100 and nothing else:
{
  "scores": {
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
  }
}
```

**Initial thresholds (hardcoded before dev tuning):**

| Label | Initial Threshold | Rationale |
|-------|:-----------------:|-----------|
| `toxic` | 25 | Same as M1 to M3; high-recall anchor label |
| `severe_toxic` | 35 | Lowered from M1 to M3's 75, as the improved prompt calibrates scores better, so a lower gate captures more true positives |
| `obscene` | 30 | Slightly lowered from 40 to recover borderline cases |
| `threat` | 45 | Raised from 40, as the boundary-example prompt reduces false positives so a higher gate is safe |
| `insult` | 45 | Slightly lowered from 50 |
| `identity_hate` | 35 | Lowered from 55 to catch coded and indirect hate speech |

**Final thresholds (dev-tuned):** Optimised per label on held-out dev set with recall floors (`severe_toxic` >= 0.50, `threat` >= 0.50, `identity_hate` >= 0.60). Saved to `threshold_from_dev_v5_cl1.json`. These override the initial values above at evaluation time. The most consequential outcome of tuning was `severe_toxic` dropping from an initial 35 to a final value of 20, which directly drove recall improvement on that label from 0.529 (M3) to 0.702 (M4).

## 5. Method 4: Severity Check Prompt (Second Pass)

**File:** `METHOD_4_RAG_HybridAugmentation.ipynb`  
**Trigger:** Called when `toxic >= 40` AND `20 <= severe_toxic <= 65` in the main prompt output  
**Purpose:** Resolves borderline `severe_toxic` cases using a focused 3-point severity rubric  
**Merge:** Final score = `0.5 x original severe_toxic + 0.5 x severity check score`

```
You are evaluating whether a toxic comment reaches the level of SEVERE toxicity.

Severity scale:
- NOT severe: single profanity, mild rudeness, casual insult, disagreement expressed rudely
- BORDERLINE: aggressive hostility, multiple strong profanities, demeaning personal attack
- SEVERE: sustained extreme abuse, dehumanising language, intense degrading attack, violent hostility,
  or repeated extreme obscenity directed at a person

Comment: {comment_text}

Rate the severity as a single integer 0-100 where:
0-30 = not severe, 30-60 = borderline, 60-100 = clearly severe.

Output ONLY this JSON:
{"severe_toxic_score": 0}
```

## 6. Method 4: Identity Hate Check Prompt (Second Pass)

**File:** `METHOD_4_RAG_HybridAugmentation.ipynb`  
**Trigger:** Called when `identity_hate` hint >= 0.08 AND `15 <= identity_hate score <= 55` in the main prompt output  
**Purpose:** Resolves borderline `identity_hate` cases by explicitly probing for group-based targeting  
**Merge:** Final score = `0.5 x original identity_hate + 0.5 x identity check score`

> **Note on "hint":** The `identity_hate hint` is a float (0.0 to 1.0) derived from the label distribution of the retrieved nearest-neighbour pool examples for the target comment. A hint of 0.08 means at least 8% of the closest pool examples carry the `identity_hate` label. This signal is boosted when the target comment contains group-identity nouns (e.g., "muslim", "immigrant", "gay") from a curated keyword list (see Appendix B).

```
Does this comment attack, demean, or express hatred toward people BECAUSE OF
their membership in a protected group (race, religion, ethnicity, gender, sexual orientation,
nationality, disability)?

Look for:
- Explicit slurs targeting a group
- Stereotyping or generalising negatively about a group
- Coded language like "these people", "go back to your country"
- Dehumanising comparisons applied to a group

Comment: {comment_text}

Rate the likelihood of identity-based hate as a single integer 0-100.

Output ONLY this JSON:
{"identity_hate_score": 0}
```

## 7. Method 4: Self-Consistency Pass

**File:** `METHOD_4_RAG_HybridAugmentation.ipynb`  
**Trigger:** Called when any of the hard labels (`severe_toxic`, `threat`, `identity_hate`) has a score in the borderline zone: `25 <= score <= 75`  
**Purpose:** Reduces variance on ambiguous cases by re-prompting with a slightly different framing and averaging the two passes  
**Merge:** Final score = `0.6 x Pass 1 + 0.4 x Pass 2` (per label)

The self-consistency pass re-uses the exact same prompt template from Section 4, with one change, where the opening line is replaced:

| Pass | Opening line |
|------|-------------|
| Pass 1 (original) | `You are an expert multi-label toxicity classifier.` |
| Pass 2 (re-evaluate) | `Re-evaluate this comment carefully, paying extra attention to severity and context:` |

All other prompt content (label definitions, boundary examples, decision rules, retrieved examples, output format) remains identical. The same `seed + 1` is used to introduce slight variation in the GPT response.

**Design rationale:** Self-consistency is restricted to the three hard labels because these are the labels with highest variance and lowest base rate. Running a second pass for all six labels would double cost with marginal benefit on already-stable labels like `toxic` and `obscene`. In practice, the second pass was triggered for approximately 25% of comments on the test split, targeting only cases where the model was genuinely uncertain.

## 8. Method 4: Skip-Gate Logic (No Prompt)

**File:** `METHOD_4_RAG_HybridAugmentation.ipynb`  
**Purpose:** Avoids unnecessary GPT calls for comments that are almost certainly clean, saving cost and latency  
**Output when triggered:** All labels set to 0, no GPT call made, row logged as `skipped_gpt_high_confidence_clean`

The skip-gate fires when all of the following conditions are met simultaneously. The neighbour check operates on the default retrieval pool of 6 neighbours (`RETRIEVE_K = 6`), not the expanded hard-label pool of 8 (`RETRIEVE_K_HIGH = 8`), which is only activated when hard-label hints are detected.

| Condition | Parameter | Value |
|-----------|-----------|-------|
| Comment is long enough to reliably assess | `SHORT_COMMENT_WORDS` | > 12 words |
| No hard-label hints from pool neighbours | `HARD_LABEL_HINT_FLOOR` | all < 0.08 |
| No soft-label hints (insult, obscene) | `SOFT_GUARD_FLOOR` | all < 0.04 |
| No high obscenity and insult combo | none | not (obscene >= 0.12 AND insult >= 0.08) |
| No sarcasm cue detected | none | sarcasm = 0 |
| No group-identity nouns present | `GROUP_IDENTITY_NOUNS` list | none found |
| No toxic keywords present | `TOXIC_KEYWORDS` list | none found |
| Top-6 neighbours mostly clean | `SKIP_CLEAN_RATIO` | >= 0.95 |
| All hint scores near zero | `SKIP_ALL_HINT_FLOOR` | all < 0.05 |

If any condition fails, the comment proceeds to GPT classification as normal.

**Design rationale:** In the full Jigsaw dataset, roughly 90% of comments are non-toxic, though the evaluation set used in this study has a deliberate 50/50 split. Skipping the most confidently clean cases targets an observed bypass rate of approximately 1.7% on the test split (69 of 4,001 comments), saving approximately $0.13 in API cost without affecting classification quality on toxic comments. The multiple override guards (toxic keywords, group nouns, sarcasm, hint combos) ensure that ambiguous or subtly toxic comments are never skipped.

## 9. Method 4: Post-Hoc Calibration Rules

**File:** `METHOD_4_RAG_HybridAugmentation.ipynb`  
**Purpose:** Enforces logical consistency between predicted labels after GPT scoring and threshold application  
**Toggle:** `APPLY_HEURISTIC_RULES = True`

These rules are applied after threshold conversion (scores to binary predictions) and after the severity and identity second passes. They do not change confidence scores, only final binary predictions, with one exception in R4.

| Rule | Logic | Rationale |
|------|-------|-----------|
| **R1: Label hierarchy** | If any of `severe_toxic`, `threat`, `identity_hate`, `insult`, `obscene` = 1, then force `toxic` = 1 | Any sub-label logically implies toxicity |
| **R2: Rare-label suppression** | If `toxic` score < 15, then force `severe_toxic`, `threat`, `identity_hate` = 0 | Prevents impossible predictions (rare label without base toxicity) |
| **R3: Insult smoothing** | If `insult` = 1 but `toxic` score < 20, then flip `insult` = 0 | Catches hallucinated insult predictions on near-clean comments |
| **R4: Severity floor** | If `obscene` score >= 80 AND `insult` score >= 70 AND `severe_toxic` score < 40, then boost `severe_toxic` score to 40 | Intense obscene abuse should register as at least borderline severe (this is the one rule that modifies scores) |
| **R5: Identity to insult hierarchy** | If `identity_hate` = 1, then force `insult` = 1 | Identity-based hate attacks are inherently insulting |

**Application order:** R4, R1, R2, R3, R5

## 10. Method 4: Initial vs Dev-Tuned Thresholds

Method 4 uses a two-stage threshold strategy.

The first stage uses **initial thresholds** (hardcoded in `LABEL_THRESHOLDS`), used during development and as the starting point for the dev-set optimiser. These differ from Methods 1 to 3 because the improved prompt changes the score distribution.

The second stage uses **dev-tuned thresholds** (saved to `threshold_from_dev_v5_cl1.json`), optimised per label on the held-out dev split using grid search with recall floor constraints. These override the initial values at evaluation time.

| Label | M1 to M3 Threshold | M4 Initial | M4 Dev-Tuned | Recall Floor Constraint |
|-------|:-------------------:|:----------:|:------------:|:-----------------------:|
| `toxic` | 25 | 25 | (from JSON) | none |
| `severe_toxic` | 75 | 35 | 20 | >= 0.50 |
| `obscene` | 40 | 30 | (from JSON) | none |
| `threat` | 40 | 45 | 16 | >= 0.50 |
| `insult` | 50 | 45 | (from JSON) | none |
| `identity_hate` | 55 | 35 | 2 | >= 0.60 |

> **Why the initial thresholds differ:** Methods 1 to 3 use a shared static threshold set because their score distributions are broadly similar. Method 4's improved prompt (severity rubric, boundary examples, second passes) shifts the distribution, particularly for `severe_toxic` and `identity_hate`, making a different starting point necessary. The effect of threshold tuning on classification performance is discussed in the report (Section 3.2.4).

## 11. Design Rationale and Iteration Log

This section documents the key design decisions made across prompt iterations and explains why each choice was made. This addresses the rubric requirement to explain why design choices worked or failed, not just report results.

### 11.1 Zero-Shot to Few-Shot: Why Static Examples Hurt Performance

Method 2 added 41 static examples to the zero-shot prompt. Contrary to expectations, this degraded macro F1 from 0.773 to 0.723. Three failure mechanisms were identified.

Context-length dilution: the 41-example block added approximately 3,000 tokens per prompt (average total 3,572 tokens), impairing the model's attention to the target comment, consistent with findings from Liu et al. (2024) on the "lost in the middle" effect in long contexts.

Example-anchoring bias: only 3 of 41 examples carried the `severe_toxic` label, biasing the model toward conservatism on that label and producing the recall collapse visible in Table 3.4 of the report (recall = 0.214).

Static mismatch: every comment received identical examples regardless of content, limiting calibration value and producing the ROC-AUC decline from 0.895 (M1) to 0.844 (M2).

**Lesson:** The number of examples is less important than their relevance to the specific comment being classified.

### 11.2 Few-Shot to RAG Dynamic: Addressing the Three Failure Mechanisms

Method 3 replaced static examples with per-query retrieval (`METHOD3_RAGDYNAMIC.ipynb`), directly addressing all three Method 2 failures. The final configuration uses `RAG_K = 7` total retrieved examples, with `RAG_HARD_K = 3` slots guaranteed for hard-minority-label positives. A cosine similarity floor of `RAG_SIM_FLOOR = 0.10` filters out weakly related examples.

Shorter context (7 vs 41 examples) brought average prompt length down to approximately 1,573 tokens (56% shorter than M2). Semantic relevance per query via `text-embedding-3-small` cosine similarity, and a naturally adapted label distribution per comment, addressed each of M2's failure modes directly.

Result: macro F1 recovered to 0.775 and ROC-AUC improved to 0.930, a gain of 3.45 percentage points over M1 on ROC-AUC.

### 11.3 RAG Dynamic to HA-RAG: Targeted Error-Driven Fixes

Method 4 (`METHOD_4_RAG_HybridAugmentation.ipynb`) addressed three specific weaknesses identified through error analysis on the Method 3 dev set.

| Weakness | Fix Applied | Effect |
|----------|-------------|--------|
| `severe_toxic` under-detection (threshold=35 too strict for M4's calibrated scores) | 4-point severity rubric and dedicated second-pass severity check | Recall: 0.529 to 0.702 |
| `threat` over-prediction (anger flagged as threat) | Explicit IS / IS NOT boundary examples and target-proximity requirement | FP rate reduced approximately 75% relative to M1 and M3 (74 FP on 4,001 rows vs 300 on 8,000 in M1; Section 3.3.2 of the report) |
| `identity_hate` under-detection (coded language missed) | Group-noun detection heuristic and dedicated second-pass identity check | Recall improvement on borderline cases |

### 11.4 Output Format Decision: Confidence Scores vs Binary Labels

Methods 2 to 4 were designed to return confidence scores (0 to 100) rather than binary labels (0/1) for two reasons.

Threshold flexibility: returning scores allows per-label threshold tuning on a dev set without re-running inference. This was critical for Method 4, where the `severe_toxic` threshold was lowered from 75 (the fixed value used in Methods 1 to 3) to an initial value of 35 when M4's improved prompt was introduced, and then further to 20 after dev-set tuning — a change that directly drove the recall improvement from 0.529 (M3) to 0.702 (M4).

ROC-AUC computation: continuous scores enable ROC-AUC reporting, providing a threshold-independent performance metric that validates whether degradation is real or just a threshold artefact. Note that ROC-AUC was not computed for Method 4 in the same way as Methods 1 to 3 because the dev/test split changes the score distribution, making direct comparison unreliable.

Method 1 requests binary 0/1 labels directly. The pipeline rescales these to 0/100 before applying thresholds, so Method 1 cannot benefit from threshold tuning in the same way, as a label is either 0 or 100 with no intermediate signal.

### 11.5 Self-Consistency: Why a Weighted Average, Not Majority Vote

The self-consistency pass (Section 7) uses a weighted average (0.6/0.4) rather than a simple majority vote or equal-weight average. Pass 1 is better calibrated because it sees the standard prompt framing, so giving it 60% weight preserves calibration while still allowing Pass 2 to correct edge cases. Majority vote requires three or more passes to be meaningful, which would roughly triple cost. Two-pass weighted averaging achieves comparable variance reduction at lower cost. The pass is also restricted to hard labels only (`severe_toxic`, `threat`, `identity_hate`) because these are the only labels where borderline scores are common enough to justify the extra GPT call.

### 11.6 Skip-Gate: Cost and Recall Trade-off

The skip-gate (Section 8) was introduced after observing that approximately 90% of comments in the full Jigsaw dataset are non-toxic. Running GPT inference on obviously clean comments wastes budget. The skip-gate requires multiple simultaneous safety conditions to be met before bypassing GPT and achieved a 1.7% bypass rate (69 of 4,001 test comments), saving approximately $0.13. The extensive override list (toxic keywords, group nouns, sarcasm detection, combo hints) ensures recall is not sacrificed, as any ambiguous signal forces the comment back into the GPT path.

### 11.7 Fallback Prompt

All methods include a fallback for cases where the main prompt returns unparseable output. Note that the fallback uses `{comment_text}` as the placeholder variable name, which is consistent with the second-pass prompts in Sections 5 and 6, while the main prompts in Sections 1 to 4 use `{comment}`. Both variable names resolve correctly in the respective `build_prompt_fn` functions in each notebook.

```
Classify this comment for toxicity. Return ONLY valid JSON with integer scores 0-100:
{comment_text}

{"scores":{"toxic":0,"severe_toxic":0,"obscene":0,"threat":0,"insult":0,"identity_hate":0}}
```

This ensures every row yields a valid prediction. Failed parses are logged to `error_log.csv` for analysis.

## Appendix A: Method 4 Retrieval Configuration

All values below are taken directly from the CONFIG section of `METHOD_4_RAG_HybridAugmentation.ipynb`.

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `RETRIEVE_K` | 6 | Default number of retrieved examples per comment |
| `RETRIEVE_K_HIGH` | 8 | Number of examples when hard-label hints are detected |
| `LOCAL_CENTROID_K` | 50 | Neighbours used to compute local centroid for purity ranking |
| `PURITY_FLOOR` (severe_toxic) | 0.45 | Minimum purity score for severe_toxic pool examples |
| `PURITY_FLOOR` (threat) | 0.55 | Minimum purity score for threat pool examples |
| `PURITY_FLOOR` (identity_hate) | 0.50 | Minimum purity score for identity_hate pool examples |
| Embedding model | `text-embedding-3-small` | Used for both pool indexing and query embedding |

## Appendix B: Group-Identity Noun List

Used by the skip-gate (Section 8) and the identity-hate hint booster. If any of these tokens appear in the comment text, the skip-gate is disabled and the `identity_hate` hint receives a boost before the main classification call.

muslim, muslims, islam, islamic, jew, jews, jewish, black, blacks, white, whites, asian, asians, chinese, mexican, mexicans, immigrant, immigrants, refugee, refugees, gay, gays, lesbian, lesbians, trans, transgender, lgbtq, women, woman, feminist, feminists, christian, christians, hindu, hindus, sikh, sikhs, arab, arabs, african, hispanic, latino, latina, native, indigenous
