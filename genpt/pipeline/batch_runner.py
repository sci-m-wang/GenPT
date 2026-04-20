"""
Batched vLLM Pipeline for GenPT Interpretation + Diagnosis
==========================================================

This module delivers the high-throughput version of Stage 2 (Interpreter) and
Stage 3 (Diagnostician) on top of :class:`~genpt.llm.vllm_client.VLLMTextClient`.

Design principles
-----------------
1. **Batch everything**.  For a single persona, Stage 2 generates *3* batched
   LLM calls (TAT × 8, Rorschach encoding × 1, SCT × 20) and Stage 3 generates
   *k* diagnosis calls (1 per applicable task).  When running on a corpus of
   personas we stack all of these into a single ``generate_batch`` call so
   that vLLM's continuous batching yields close to peak GPU utilisation.

2. **Rich CoT prompts**.  The original prompts were terse and JSON-only.
   The new prompts invite the model to reason step-by-step (either via
   Qwen3 thinking mode or an explicit ``<analysis>`` block) and then emit a
   strict ``<answer>`` JSON block, giving us (a) much better accuracy and
   (b) training signal we can reuse as rationale-backfill targets.

3. **Keep old classes working**.  The original
   :class:`~genpt.pipeline.interpreter.Interpreter` and
   :class:`~genpt.pipeline.diagnostician.Diagnostician` are retained for
   single-persona evaluation.  The new ``BatchInterpreter`` /
   ``BatchDiagnostician`` here are additive, not replacements.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import (
    SCORS_G_DIMENSIONS,
    SCT_SUBCONSTRUCT_TO_DOMAIN,
)
from ..llm.base import GenerationConfig
from ..llm.vllm_client import VLLMTextClient
from .examinee import RorschachResponse, SCTResponse, TATResponse
from .interpreter import InterpretationResult, SCORSGScore, SCTScore, SRASScore
from .diagnostician import (
    BigFivePrediction,
    DepressionPrediction,
    DiagnosisResult,
    MBTIPrediction,
    SuicidePrediction,
)

logger = logging.getLogger("genpt.pipeline.batch")


# ============================================================
# Prompt templates (carefully tuned)
# ============================================================

SCORS_G_SYSTEM = """You are a senior clinical psychologist certified in the SCORS-G (Social Cognition and Object Relations Scale — Global Rating Method) coding system.

You will read a single TAT narrative and assess it on all 8 SCORS-G dimensions.
Each dimension is rated on a 1–7 scale where 1 is the most pathological end
and 7 is the healthiest end.  Use integer scores and anchor every score in
at least one concrete piece of textual evidence from the narrative.

Dimensions (use these exact 3-letter codes in your output):
  COM — Complexity of Representations of People
  AFF — Affective Quality of Representations
  EIR — Emotional Investment in Relationships
  EIM — Emotional Investment in Values & Moral Standards
  SC  — Understanding of Social Causality
  AGG — Experience & Management of Aggressive Impulses
  SE  — Self-Esteem
  ICS — Identity & Coherence of Self

Anchors (abbreviated):
  1–2: severely impaired / gross distortions / no mentalisation
  3  : boundary / simplistic / concrete
  4  : adequate but limited
  5  : clear strengths
  6–7: rich, nuanced, well-integrated

Procedure:
  1. Read the narrative in full.
  2. For each dimension, identify the evidence, then assign the integer
     score that matches the SCORS-G anchors.
  3. When evidence is absent for a dimension, default to 4 (neutral) and say
     so.
  4. Do NOT produce consecutive identical scores unless the evidence is
     truly identical — differentiate the dimensions.

Output format (EXACTLY — no extra prose outside the tags):
<analysis>
COM: <1-3 sentences of evidence → score>
AFF: ...
EIR: ...
EIM: ...
SC : ...
AGG: ...
SE : ...
ICS: ...
</analysis>
<answer>
{"COM": <int>, "AFF": <int>, "EIR": <int>, "EIM": <int>, "SC": <int>, "AGG": <int>, "SE": <int>, "ICS": <int>}
</answer>"""


SCORS_G_USER_TMPL = """TAT Card: {image_id}

Narrative:
\"\"\"
{narrative}
\"\"\"

Produce the SCORS-G coding now."""


SRAS_SYSTEM = """You are an expert Rorschach coder.  You will encode a set of Rorschach card responses into the Simplified Rorschach Analysis System (SRAS) variables.

Rules:
  * Count only features that are CLEARLY present in the text; when in doubt, do NOT code the variable.
  * Produce integer counts (0 if absent).
  * Think about each response once, then aggregate counts across all responses.

Variables to count (use exactly these keys):
  P, FQo, FQu, FQ-, WSumCog,
  FC, CF, C, C', Y, V, T,
  M, FM, m,
  COP, AG, MOR, AGC, AGM,
  H, M-

Output format (strict):
<analysis>
<1-2 short paragraphs summarising what drove the counts; cite specific cards>
</analysis>
<answer>
{"P": n, "FQo": n, "FQu": n, "FQ-": n, "WSumCog": n, "FC": n, "CF": n, "C": n, "C'": n, "Y": n, "V": n, "T": n, "M": n, "FM": n, "m": n, "COP": n, "AG": n, "MOR": n, "AGC": n, "AGM": n, "H": n, "M-": n}
</answer>"""


SRAS_USER_TMPL = """Rorschach responses from one testee:

{cards_block}

Provide SRAS encoding now."""


SCT_SYSTEM = """You are a clinical psychologist scoring Sentence Completion Test (SCT) items using the Rotter-style 0–6 scale:

  0–1: very positive / well-adjusted / resilient
  2  : mildly positive
  3  : neutral / conventional
  4  : mildly conflicted
  5  : clearly conflicted
  6  : severely conflicted / maladaptive

Only use the stem + the testee's completion; do NOT read between the lines beyond what is textually supported.

Output format (strict, one answer per stem):
<analysis>
[<stem_id>] <brief evidence → score>
...
</analysis>
<answer>
{"<stem_id_1>": <0-6>, "<stem_id_2>": <0-6>, ...}
</answer>"""


SCT_USER_TMPL = """Score every SCT item below.  Use the stem id as the JSON key.

{items_block}"""


# ---- Interpreter TEACHER (for Phase D1 cold-start SFT) ----
#
# Distinct from the production SCORS/SRAS/SCT prompts above.  The teacher
# variant consumes the *raw* behaviors (all TAT narratives + all Rorschach
# perceptions + all SCT completions) for a single behaviour group and is told
# the persona's ground-truth psych-state labels.  It must emit a single
# chain-of-thought that links behaviours → constructs → scores, then emit a
# consolidated JSON block with the full interpretation.  This teaches the
# production interpreter the full *behavior → structured-score → state*
# mapping as a single reasoning chain (not per-prompt as the zero-shot path).

INTERP_TEACHER_SYSTEM = """You are a senior clinical psychologist specialised in projective tests.

You will receive a subject's *complete* raw responses to a TAT/Rorschach/SCT battery (1 behaviour group) together with that subject's validated psychological labels (Big Five / MBTI / depression / suicide — whichever are available for this subject).

==> TEACHING MODE <==
You have privileged access to the labels.  Use them only as an internal
coherence check; do not copy them into your output.  Your job is to:

  1. Inside <think>...</think>, reason aloud across the behaviours.  Walk
     through each TAT narrative (one by one), each Rorschach card, and the
     SCT themes.  For each, identify the *specific* behaviour cue (quote or
     paraphrase briefly) and the *construct* it implicates (SCORS-G
     dimension, SRAS domain, SCT domain), then the numeric score and why.
     Aim for convergent-validity across modalities and for scores whose
     aggregate pattern is *consistent* with the provided labels
     (e.g. high depression → low SCORS AFF/SE, high SCT ER;  introvert MBTI
     → low SCORS EIR, elevated SCT IR conflict).

  2. After </think>, output a single JSON object with the full
     interpretation, using this schema (omit dimensions you could not
     score):

     {{
       "scors_g": {{
         "<image_id>": {{"COM": 1..7, "AFF": 1..7, "EIR": 1..7,
                        "EIM": 1..7, "SC": 1..7, "AGG": 1..7,
                        "SE":  1..7, "ICS": 1..7}},
         ...
       }},
       "sras": {{"CPS": 1..7, "ARS": 1..7, "IRS": 1..7, "SCS": 1..7}},
       "sct":  {{"FA": 0..6, "CA": 0..6, "SA": 0..6, "IR": 0..6, "ER": 0..6}}
     }}

Use the exact TAT image_id strings present in the input.  Every TAT image
must appear as a key in "scors_g".  All four SRAS and all five SCT domains
must be scored.  Do not emit any text after the closing brace.

Calibration reminder:
  - SCORS-G: 1 = most pathological / impoverished, 4 = neutral, 7 = most adaptive.
  - SRAS:    1 = weakest functioning on that domain, 7 = strongest.
  - SCT:     0 = well-adjusted / positive, 6 = severely conflicted.
"""


INTERP_TEACHER_USER_TMPL = """=== TAT NARRATIVES ===
{tat_block}

=== RORSCHACH RESPONSES ===
{rorschach_block}

=== SCT COMPLETIONS ===
{sct_block}

=== PRIVILEGED PSYCH-STATE LABELS (for your internal coherence check only) ===
{labels_block}

Emit <think>...</think>, then a single JSON object with the full
interpretation, in the schema specified by the system prompt.
Do not mention the labels inside <think> — only your scoring logic.
"""


# Production variant — used at SFT TIME as the user prompt (GT labels stripped).
# The assistant target is the teacher-produced <think>...</think>{JSON} string.
INTERP_PRODUCTION_USER_TMPL = """=== TAT NARRATIVES ===
{tat_block}

=== RORSCHACH RESPONSES ===
{rorschach_block}

=== SCT COMPLETIONS ===
{sct_block}

Reason step-by-step inside <think>...</think>, then emit a single JSON object
with the full interpretation (scors_g / sras / sct), following the schema
specified by the system prompt.
"""


# ---- Diagnostician ----

DIAG_SYSTEM_BASE = """You are an expert clinical psychologist serving as the diagnostician in a multi-stage projective-test-based assessment.

You will receive *structured interpretation scores* produced by a coder (TAT via SCORS-G, Rorschach via SRAS, SCT via domain scores) together with their rationales.  Your job is to integrate these indicators into a final prediction.

Guidelines:
  * Consider TAT, Rorschach, and SCT in concert.  Each is informative but noisy.
  * SCORS-G:  AFF/EIR/SE are most informative for mood / self-attitudes;
              COM/SC for cognitive style; EIM for super-ego / conscientiousness;
              AGG for externalising features.
  * SRAS:  CPS → cognitive processing fidelity, ARS → affect modulation,
           IRS → interpersonal representation quality, SCS → coping balance
           (EA − es).  Strongly negative SCS or IRS warrants caution.
  * SCT domains:  FA (Family), CA (Career), SA (Self-Attitudes), IR
           (Interpersonal), ER (Emotion Regulation).  Values 0–6, higher =
           more conflicted.

Calibration principles (critical):
  * AVOID prediction collapse.  Do NOT default to the most severe category
    (e.g. Depression=3, Suicide=2) as a "safe" answer.  Base your choice on
    actual evidence strength.
  * Use the full scale.  For ordinal scales (0–3 or 1–5), each level has a
    distinct meaning; pick the level whose anchors best match the evidence.
  * Absence of positive content ≠ presence of severe pathology.  If a
    person's projective output is neutral, the correct label is typically
    a mid or low level, not the extreme.
  * Be willing to predict 0 (none) for depression/suicide when no
    indicators are present — this is common and correct in a balanced sample.

Reason step-by-step inside <analysis>.  Cite the specific indicators
(e.g. "AFF=3, low", "MOR present on TAT #2") that drove your decision.  Then
produce the required answer.  The answer line must be machine-parsable and
match the exact format specified in the user prompt."""


DIAG_PROMPTS_BY_VERSION: Dict[str, Dict[str, str]] = {
    "big_five": {},
    "mbti": {},
    "depression": {},
    "suicide": {},
}


DIAG_PROMPTS_BY_VERSION["big_five"][
    "v4"
] = """Predict the Big Five personality levels on a 1–5 scale.

CRITICAL CONTEXT: The projective tests (TAT, Rorschach, SCT) inherently
surface affective/unconscious material, which makes every respondent look
more anxious, self-critical, and conflicted than they are in daily life.
This creates systematic prediction biases you must compensate for:

  * DO NOT let projective-test negativity inflate Neuroticism.  A character
    who responds neutrally to everyday situations outside the test is N=2
    or N=3 even if they produce some negative TAT content.
  * DO NOT default Openness to 4 just because the character produces
    narrative TAT responses — that's a task demand, not a trait marker.
  * DO NOT default Extraversion to 3 for unclear cases.  Extraversion
    should reflect the character's actual social energy from the SCT IR
    domain and TAT interpersonal narratives.

Behavioural anchors (anchor to the CHARACTER's description in the system
prompt, not to the projective-test tone):

  O (Openness to experience)
    1 = strictly conventional, rejects new ideas, narrow concrete focus
    2 = pragmatic, prefers familiar, occasional curiosity
    3 = balanced, open to variety within familiar domains
    4 = imaginative, intellectually curious, artistic
    5 = unconventional, deeply creative, actively seeks novelty
    * Anchor on SCT creativity/career breadth, NOT on TAT narrativity.

  C (Conscientiousness)
    1 = impulsive, chaotic, unreliable, abandons goals
    2 = easily distracted, inconsistent follow-through
    3 = averagely organised, meets basic obligations
    4 = disciplined, goal-directed, dependable
    5 = meticulous, highly self-controlled, perfectionistic
    * If SCT CA shows career avoidance or chaotic self-management → 1-2.
    * Do NOT default to 3 for every character.

  E (Extraversion)
    1 = very withdrawn, drained by any social contact
    2 = reserved, prefers solitude, quiet
    3 = balanced, adapts to both contexts
    4 = outgoing, talkative, energised by interaction
    5 = highly assertive, thrill-seeking, dominant in groups
    * Anchor on SCT IR tone + TAT interpersonal scenes.
    * If the character's SCT IR is warm+active → likely 4.  If SCT IR is
      conflicted+avoidant → 1-2.  Resist the pull to 3.

  A (Agreeableness)
    1 = antagonistic, cynical, cold, competitive
    2 = blunt, sceptical, conditional cooperation
    3 = balanced, cooperative when convenient
    4 = warm, trusting, altruistic
    5 = deeply empathetic, selfless, avoids conflict
    * AGG (SCORS-G) high + cynical SCT IR → 1-2.
    * Warm caregiving narratives → 4-5.

  N (Neuroticism) — THE MOST OVER-PREDICTED DIMENSION
    1 = emotionally very stable, calm under stress, composed
    2 = generally calm, mild occasional stress reactions (THIS IS THE MODAL
        LEVEL for typical characters — do not over-escalate)
    3 = average emotional reactivity, occasional worry
    4 = frequently anxious, moody, self-doubting (requires PERVASIVE
        instability, not just some AFF negativity)
    5 = chronically volatile, severe mood swings, overwhelming anxiety
    * N=4 is NOT justified solely by low AFF on some TAT cards; it requires
      pervasive SE instability AND anxiety themes AND self-doubt across
      multiple projective channels.
    * Fictional characters who are "cool", "composed", or "stoic" are N=1
      or N=2 regardless of their dramatic backstory.

Calibration:
  * Use the full 1–5 range across the five dimensions.  A healthy person
    often has a mix like O=4, C=3, E=3, A=4, N=2, NOT 3/3/3/3/3.
  * If you wrote "moderate" for every dimension, reconsider — rarely is
    a well-defined character average on all five.
  * Ground each dimension in SPECIFIC projective evidence, not generic
    psychological vibes.

In <analysis>, for each of O/C/E/A/N, cite 1-2 concrete indicators (a
specific TAT card, SCT item ID, SRAS score) and then state the level.

Answer format (EXACT, one line):
<answer>O=<1-5>, C=<1-5>, E=<1-5>, A=<1-5>, N=<1-5></answer>"""

DIAG_PROMPTS_BY_VERSION["mbti"][
    "v2"
] = """Predict the MBTI personality type as a 4-letter code (e.g. INFP).

Axis anchors:
  1. E vs I (Energy):
       E = externally oriented, draws energy from interaction, thinks aloud
       I = internally oriented, drained by social contact, reflects first
  2. S vs N (Information):
       S = concrete, detail-focused, trusts experience
       N = abstract, pattern-focused, trusts intuition/possibility
  3. T vs F (Decision):
       T = logic-first, objective criteria, direct feedback
       F = values-first, empathy-driven, harmony-seeking
  4. J vs P (Lifestyle):
       J = planned, decisive, prefers closure
       P = flexible, spontaneous, keeps options open

Calibration:
  * Score each axis independently; do NOT let one letter bias another.
  * J/P is the most commonly mis-scored axis.  A person who improvises,
    resists early commitment, or tolerates ambiguity is P even if they are
    otherwise disciplined on the task.
  * Creative / unconventional people are usually N, not S.

In <analysis>, discuss each axis briefly (1\u20132 sentences with a concrete
indicator) before the answer line.

Answer format (EXACT):
<answer>XXXX</answer>"""


# V3: designed for fictional characters (CharacterRAG).  Ranks the canonical
# description ABOVE projective sub-scores, gives J/P its own behavioural
# anchors (was 0.53 accuracy in V2 \u2014 barely above chance), and demands
# per-axis evidence citation to block "one letter leaks into another".
DIAG_PROMPTS_BY_VERSION["mbti"][
    "v3"
] = """Predict the MBTI personality type as a 4-letter code (e.g. INFP).

================================================================
Primary-evidence rule
================================================================
The "Character Description" block (if present) is the PRIMARY source for
MBTI axes.  Community MBTI typings are almost always based on canonical
behaviour, not on subtle projective patterns.  The projective block is a
secondary / confirming signal only.

================================================================
Axis-by-axis decision procedure
================================================================
Work one axis at a time.  For each axis, cite ONE canonical behaviour
(or, failing that, ONE specific projective indicator) and then commit to
a letter.  Do NOT let axis-1 bias axis-2, etc.

1) E vs I  (where do they get energy?)
     E: talks to think, initiates contact, energised by crowds,
        publicly expressive, uncomfortable alone for long.
     I: thinks before speaking, drained by small talk, needs alone time
        to recover, rich inner life, small chosen circle.
   Typical story cues: E = "life of the party", "takes the lead in
   group scenes"; I = "observer", "retreats to recharge".

2) S vs N  (how do they process information?)
     S: notices concrete detail first, prefers what is tested and
        practical, distrusts wild speculation, lives in the present.
     N: sees patterns and abstractions first, jumps to big-picture
        implications, loves theoretical/hypothetical framings.
   Story cues: S = "methodical craftsperson", "grounded in routine";
   N = "visionary", "obsessed with an idea/goal most call impossible",
   chuunibyou / fantasist characters are almost always N.

3) T vs F  (how do they decide?)
     T: logic, justice, fairness, impersonal truth, accepts harsh
        feedback, cuts through emotion to find the correct answer.
     F: harmony, care for specific people, values the felt impact of
        a decision, protects others' dignity, often diplomatic.
   Story cues: T = "brutally honest", "principled to the point of
   coldness"; F = "kind", "loyal to her friends", "sacrifices logic
   for the people she loves".  A character with a famous code of
   ethics built on PEOPLE = F; a code built on ABSTRACT PRINCIPLES
   like justice/power = T.

4) J vs P  (how do they handle the outer world?) \u2014 MOST OFTEN MISSED.
     J = prefers closure, decides early, plans schedules, uncomfortable
         with open loops, "settled" lifestyle.  Typical J cues:
           * \"has a clear long-term goal and sticks to it\";
           * \"famously organised / punctual\";
           * \"cannot rest until the task is finished\";
           * shounen heroes with a fixed mission arc are usually J.
     P = keeps options open, improvises, resists commitment, comfortable
         with ambiguity, reacts to situations rather than pre-planning.
         Typical P cues:
           * \"laid-back\", \"goes with the flow\";
           * \"changes tactics mid-fight\";
           * \"avoids definitive answers\";
           * saitama (no plan, just works out), goku (fights for fun) \u2192 P.
   Crucial disambiguators:
     * Being \"disciplined\" at ONE thing (training, work) does NOT make
       someone J.  A character who is disciplined at fighting but chaotic
       about everything else is P.
     * Being "adventurous" or "sociable" does NOT make someone P;
       evaluate ONLY their relationship to plans / closure.
     * Strategists who plan meticulously (Light Yagami, L) are J.
     * Improvisers who make it up as they go (Anya, Son Goku,
       Saitama, Chika) are P.

================================================================
Calibration
================================================================
  * Creative / unconventional / fantasy-prone characters are almost
    always N, not S.  Default doubt \u2192 N.
  * For famous fictional characters, lean on widely-known personality
    typings where they converge with the canonical description.
  * Warm empathetic protagonists are usually F; principled-logic
    antagonists are usually T.  Do NOT confuse "kind and competent"
    with T \u2014 kindness to specific people is F.
  * Decide each axis on ITS OWN evidence.  Do not let (e.g.) strong E
    pull J upward.

In <analysis>, write FOUR short lines, one per axis:
  EI: <cue> \u2192 <letter>
  SN: <cue> \u2192 <letter>
  TF: <cue> \u2192 <letter>
  JP: <cue> \u2192 <letter>
Then emit the answer.

Answer format (EXACT):
<answer>XXXX</answer>"""

DIAG_PROMPTS_BY_VERSION["depression"][
    "v3"
] = """Predict depression risk level on a 0–3 scale using BDI-style
anchors (severity over the past 2 weeks, inferred from the projective data).

Each level requires SPECIFIC combinations of indicators, not a generic "more
is worse":

  0 = Minimal.
      - No persistent sadness, no anhedonia, no guilt/worthlessness.
      - AFF near neutral (≥4), SE stable (≥4), SCT SA/ER ≤ 3.
      - Even if work/life stress shows in SCT, mood remains intact.

  1 = Mild.  (The most under-predicted level — read carefully.)
      - Occasional low mood, self-doubt, pessimism, but NO anhedonia.
      - Still engaged with life / relationships; can still enjoy some things.
      - Typical profile: AFF 3-4, SE 3, mild self-critical SCT (score 3-4),
        no persistent hopelessness, no death/morbid themes.
      - Strong SCT conflict in FA/CA/IR (family/career/interpersonal stress)
        WITHOUT low SE + hopelessness is MILD, not moderate.  Life stress ≠
        depression.
      - If the person shows career anxiety, family friction, or social
        avoidance but retains self-worth and some capacity for pleasure →
        level 1.

  2 = Moderate.
      - Pervasive low mood AND at least ONE of: anhedonia / self-worthlessness
        / persistent guilt.
      - AFF clearly negative (≤3), SE clearly depleted (≤3), SCT SA or ER ≥ 4,
        MOR present but not dominant.
      - Self-critical content is recurrent, not just situational.

  3 = Severe.
      - Bleak, pervasive hopelessness + marked worthlessness/guilt +
        near-total anhedonia.
      - AFF very negative (≤2), SE ≤ 2, SCS deeply negative, MOR/V dominant,
        SCT SA AND ER ≥ 5, and explicit themes of "no future", "giving up",
        possible self-harm cognition.
      - Requires MULTIPLE severe indicators simultaneously.

Calibration:
  * Do NOT default to 3.  "Severe" requires the confluence of negative mood,
    low self-worth, hopelessness, AND near-total anhedonia.  Missing any one
    of these → drop to level 2.
  * Depression=1 exists and is common.  A person with career/family conflict
    and mild self-doubt but intact self-esteem is level 1, not 2.
  * If AFF average is ≥3 and SE is ≥3, the level is almost certainly 0 or 1,
    regardless of how many SCT items score 5.

In <analysis>, explicitly walk through the four criteria (mood, self-worth,
hopelessness, anhedonia) and indicate which are present/absent.  Then map
to a level.

Answer format (EXACT):
<answer>Depression=<0-3></answer>"""

DIAG_PROMPTS_BY_VERSION["suicide"][
    "v4"
] = """Predict suicide risk level on a 0–3 scale (Columbia/BSS-style).

Each level has STRICT evidence requirements to prevent bimodal (0 or 3)
collapse.  Level 1 and 2 are common and must be used when evidence is
intermediate:

  0 = None.
      - No passive death wish, no self-harm ideation, no hopelessness.
      - No MOR/V about self, no self-destructive SCT content.
      - If you see no mention of dying, ending, or "disappearing" → 0.

  1 = Low (passive ideation).  COMMON — do not skip.
      - Passive death wish or fleeting thoughts ("wish I weren't here",
        "would be better if I disappeared", "tired of living"), WITHOUT
        active ideation.
      - Mild hopelessness, no method considered.
      - MOR ≤ 5 on 1-2 cards, occasional morbid content.
      - Any ONE of these alone → at least level 1, not 0:
          (i) any dysphoric death imagery in TAT,
          (ii) "what's the point" / "I don't matter" SCT content,
          (iii) MOR present on at least one card.

  2 = Moderate (active ideation without plan).  ALSO COMMON — use when
      evidence is strong but not extreme.
      - Active self-directed harm thoughts but NO explicit method/timeline.
      - Hopelessness + low SE (≤3) + MOR on multiple cards.
      - SCT ER with self-destructive tone but no concrete plan.
      - Phrases like "I'm a burden", "things would be better without me",
        "I can't go on" — these indicate active ideation → level 2 minimum.

  3 = High (plan/intent).  STRICTLY: require ONE of these:
      (a) Explicit plan, method, or timeline stated in any source.
      (b) Explicit intent phrases: "我该结束了", "I'm going to", "that's it
          for me", "won't see tomorrow", combined with AFF ≤ 2, SE ≤ 2,
          MOR ≥ 8, and at least 2 cards with dominant morbid themes.
      (c) Overt self-harm cognition (cutting, pills, specific self-violence
          imagery) in TAT or SCT regardless of AFF level.
      * Severe depression alone does NOT trigger level 3.
      * Hopelessness + low SE alone does NOT trigger level 3 (that's 2).
      * Morbid themes alone do NOT trigger level 3 — there must be clear
        ideation or intent targeted AT THE SELF.

Calibration:
  * Levels 1 and 2 are frequent; do NOT collapse to 0 or 3.
  * If any passive ideation indicator is present and nothing more severe
    → level 1 (not 0).
  * If active ideation present but no method/intent → level 2 (not 3).
  * Level 3 needs explicit self-directed harm cognition or specific plan.
  * Depression=3 ≠ Suicide=3.  Check suicide evidence independently.

In <analysis>, explicitly state:
  (a) Passive ideation present?  Cite specific indicator.
  (b) Active ideation / self-directed harm cognition present?  Cite.
  (c) Explicit plan/intent/self-harm cognition present?  Cite.
Based on which is the highest level of evidence, choose the matching label.

Answer format (EXACT):
<answer>Suicide=<0-3></answer>"""


# --- V3 suicide prompt (previous best: macro-F1=0.221 on label GT) ---
# V4 tightened criterion (a)(b)(c) → level-3 collapsed to 1/2 → F1 dropped
# to 0.142. V3 is the preserved best-in-class; V4 kept for reference.
DIAG_PROMPTS_BY_VERSION["suicide"][
    "v3"
] = """Predict suicide risk level on a 0–3 scale (Columbia/BSS-style).

Each level has specific evidence requirements:

  0 = None.
      - No passive death wish, no self-harm ideation, no hopelessness.
      - No MOR/V about self, no self-destructive SCT content.
      - If you see no mention of dying, ending, or "disappearing" → 0.

  1 = Low (passive ideation).
      - Passive death wish without active ideation: "wish I weren't here",
        "would be better if I disappeared", "tired of living".
      - Hopelessness present but no method considered.
      - Mild MOR/V on 1-2 cards, no dominant morbid theme.

  2 = Moderate (active ideation without plan).
      - Active ideation with self-directed harm thoughts, but no specific
        method or timeline.
      - Strong hopelessness + low SE + MOR/V on multiple cards.
      - SCT ER shows self-destructive tone without concrete plan.

  3 = High.  (Elevate to 3 when evidence is strong, even without explicit
     plan — the projective data rarely reveals plans directly.)
      - Any ONE of the following is sufficient for level 3:
        (a) Explicit plan / method / timeline mentioned in any source.
        (b) Explicit intent ("I am going to", "\u6211\u8be5\u7ed3\u675f\u4e86") combined with
            severely negative AFF (\u22642), SE \u2264 2, and MOR \u2265 8.
        (c) Multiple cards with MOR \u2265 8 or dominant death/ending themes
            AND AFF \u2264 2 AND SCS \u2264 -15 AND "no hope / \u8a72\u7d50\u675f" in TAT/SCT.
      - Projective data usually lacks explicit plan statements; do NOT
        refuse to give level 3 simply because no plan is verbalised if
        criterion (b) or (c) is met.

Calibration:
  * Level 0 is COMMON and correct when no ideation/hopelessness markers
    exist \u2014 do NOT default to 1+ out of caution.
  * Level 3 does NOT require an explicit plan.  It requires unambiguous
    self-destructive cognition with severe affective/cognitive collapse.
  * Depression severity alone is NOT sufficient for elevated suicide risk.
    Require ideation/hopelessness evidence specifically.

In <analysis>, explicitly state:
  (a) Is passive ideation present?
  (b) Is active ideation / self-harm cognition present?
  (c) Is there evidence meeting criterion (a), (b), or (c) of level 3?
Cite specific indicators (MOR scores, TAT phrases, SCT items) for each.

Answer format (EXACT):
<answer>Suicide=<0-3></answer>"""


# ---------------------------------------------------------------------------
# V5 iterations (2026-04-17) — per-task reopening of the specific failure
# modes observed on V3/V4.
# ---------------------------------------------------------------------------

# V5 Big Five: restore V1's concise anchor style (V4 got 0.200 vs V1=0.224)
# plus a single-line reminder against the two specific default biases.
DIAG_PROMPTS_BY_VERSION["big_five"][
    "v5"
] = """Predict the Big Five personality levels on a 1\u20135 scale.

Behavioural anchors (1 = strongly low, 5 = strongly high):

  O (Openness): 1 conventional / 3 balanced / 5 unconventional & inventive.
  C (Conscientiousness): 1 impulsive / 3 averagely organised / 5 meticulous.
  E (Extraversion): 1 very withdrawn / 3 ambivert / 5 highly outgoing.
  A (Agreeableness): 1 antagonistic / 3 cooperative when convenient / 5 selfless.
  N (Neuroticism): 1 very stable / 3 average reactivity / 5 chronically volatile.

Anti-bias reminders (apply to every character, not just some):
  * Openness: do NOT default to 4 just because TAT elicits narrative. The
    test induces narrativity universally; only give O=4-5 if the SCT or
    Rorschach shows genuine intellectual/artistic breadth.
  * Extraversion: do NOT default to 3 for unclear cases. Anchor on SCT IR
    tone + TAT interpersonal scenes; warm+active \u2192 4, conflicted+avoidant \u2192 1-2.
  * Neuroticism: projective tests inflate N. A composed / stoic character
    is N=2 regardless of AFF negativity on individual cards. N=4 requires
    pervasive SE instability across multiple channels.

Ground each dimension in SPECIFIC projective evidence (TAT card, SCT item,
SRAS score). In <analysis>, give one short sentence per dimension citing the
indicator that drove the level.

Answer format (EXACT, one line):
<answer>O=<1-5>, C=<1-5>, E=<1-5>, A=<1-5>, N=<1-5></answer>"""


DIAG_PROMPTS_BY_VERSION["big_five"][
    "v6"
] = """Predict the Big Five personality levels on a 1\u20135 scale for the
character described above.  Use the FULL 1\u20135 range; most well-defined
characters are extreme on at least two dimensions, NOT moderate on all five.

================================================================
Primary-evidence rule
================================================================
The "Character Description" block is the PRIMARY source for BF levels.
The projective interpretation block is a NOISY secondary signal.
If a canonical trait contradicts the projective reading, trust the canon.

Rule of thumb: predict the level a psychologist who had READ the
character's story would give, not the level that comes out of adding up
projective sub-scores.

================================================================
Dimension anchors (1 = very low, 5 = very high)
================================================================

O (Openness to experience) \u2014 unconventionality, intellectual / creative range.
  1 = strictly conventional, black-and-white thinker, rejects novelty
      (e.g. a stubborn rule-follower, a conformist athlete).
  2 = pragmatic, routine-bound, limited imagination.
  3 = balanced.
  4 = imaginative, curious, artistic within one domain.
  5 = wildly unconventional, breaks format, chuunibyou / fantastic /
      iconoclastic thinkers, characters whose defining trait IS being weird.
  Anchor: canonical trait + SCT creativity/career breadth, NOT TAT narrativity.
  WARNING: O=3 is the most OVER-predicted level. If the character is
  famous for bizarre imagination or conventional rigidity, pick 5 or 1.

C (Conscientiousness) \u2014 self-discipline, organisation, goal-pursuit.
  1 = impulsive, abandons goals mid-task (typical "chaotic" heroes).
  2 = easily distracted, inconsistent follow-through.
  3 = average.
  4 = disciplined, goal-directed (typical "genius student", elite athlete).
  5 = meticulous, perfectionistic, monk-like.
  Anchor: canonical behaviour first; SCT CA (career) only as confirmation.

E (Extraversion) \u2014 social energy, assertiveness, outward activity.
  1 = socially avoidant, drained by contact (typical shy / hikikomori).
  2 = reserved, prefers solitude.
  3 = ambivert.
  4 = outgoing, energised by social contact (typical shounen heroes).
  5 = dominant, thrill-seeking, life-of-the-party.
  Anchor: canonical behaviour; SCT IR + TAT interpersonal scenes second.

A (Agreeableness) \u2014 warmth, cooperation, altruism.
  1 = antagonistic, cold, cynical (typical villains / rivals).
  2 = blunt, conditional cooperation.
  3 = balanced.
  4 = warm, trusting, altruistic.
  5 = self-sacrificing, radical empathy.

N (Neuroticism) \u2014 emotional INSTABILITY.  THE MOST OVER-PREDICTED DIMENSION.
  1 = very stable, composed under pressure, rarely upset
      (e.g. stoic elders, calm prodigies, characters famous for equanimity).
  2 = generally calm, mild occasional stress (MODAL level for well-adjusted
      heroes \u2014 do NOT escalate).
  3 = average reactivity, occasional worry.
  4 = frequently anxious / moody / self-doubting, AND this instability is
      a defining canonical trait, NOT just projective-test artefact.
  5 = chronically volatile, severe mood swings, panic/rage.
  HARD RULES:
    * Famously stoic / cool / "I don't care" characters \u2192 N=1 or N=2,
      regardless of how dark the TAT cards make them look.
    * A character whose canonical description contains \u2265 2 of
      {"calm", "composed", "stoic", "patient", "cold", "unflappable",
       "confident"} \u2192 N \u2264 2.
    * A character with canonical panic attacks, anxiety disorder, or
      chronic self-hatred \u2192 N \u2265 4.
    * Do NOT set N \u2265 3 just because projective output is negative.
      Projective tests DESIGNED to surface unconscious affect \u2014 every
      respondent looks somewhat anxious on them.

================================================================
Calibration checklist \u2014 before you write the answer:
================================================================
  1. Identify 2 dimensions on which this character is extreme (1 or 5)
     based on canonical description.  If you can't name any, re-read.
  2. If you wrote "3" for \u2265 4 of 5 dimensions, you are wrong \u2014
     reconsider which two should be extreme.
  3. Did you predict N \u2265 3 for a character everyone calls "calm"?
     Step down to 2.
  4. Did you predict O = 3 for a character whose personality IS their
     weirdness / inventiveness?  Step up to 5.

In <analysis>, for each of O/C/E/A/N, give ONE short sentence: cite either
a canonical trait (from the character description) OR a specific projective
indicator that drove the level.

Answer format (EXACT, one line):
<answer>O=<1-5>, C=<1-5>, E=<1-5>, A=<1-5>, N=<1-5></answer>"""


# V5 Suicide: restore V3's permissive (a)/(b)/(c) level-3 pathway, and add
# concrete language examples to make level-1 and level-2 easier to spot
# (V3 collapsed to bimodal 0 vs 3; V4 over-corrected by requiring explicit
# plan and killed level-3 recall).
DIAG_PROMPTS_BY_VERSION["suicide"][
    "v5"
] = """Predict suicide risk level on a 0\u20133 scale (Columbia/BSS-style).

Each level has specific, non-overlapping evidence requirements. Use the
full 0/1/2/3 range; levels 1 and 2 are common and must NOT be skipped.

  0 = None.
      - No death wish, no self-harm ideation, no hopelessness.
      - No MOR/V about self, no self-destructive SCT content.

  1 = Low (PASSIVE ideation). Examples of level-1 content:
      - "I wish I weren\u2019t here" / "I\u2019m tired of living"
      - "It would be better if I disappeared"
      - "What\u2019s the point" / "Nothing matters anymore"
      - "I won\u2019t see another day" (passive, not a stated plan)
      - MOR content on 1\u20132 TAT cards without explicit self-harm imagery.
      Any ONE of these \u2192 at least level 1 (not 0). No method, no intent.

  2 = Moderate (ACTIVE ideation WITHOUT plan). Examples:
      - "I\u2019m a burden; they\u2019d be better off without me."
      - "I can\u2019t go on like this." / "I don\u2019t want to exist."
      - "Sometimes I think about just ending it" (no method stated)
      - Active self-directed harm thoughts + hopelessness + low SE (\u22643) +
        MOR on multiple cards, but no method/timeline stated.
      - SCT ER with self-destructive tone but no concrete plan.
      Requires self-DIRECTED harm thought content, not just morbid themes.

  3 = High (plan, intent, or unambiguous self-harm cognition). Any ONE of:
      (a) Explicit plan, method, or timeline stated in any source.
      (b) Explicit intent phrases ("I\u2019m going to end it", "\u6211\u8be5\u7ed3\u675f\u4e86",
          "won\u2019t see tomorrow") COMBINED with AFF \u2264 2, SE \u2264 2, and
          MOR \u2265 8.
      (c) Overt self-harm cognition imagery (cutting, pills, jumping,
          specific self-violence) in TAT or SCT content, regardless of
          AFF level.
      Projective data rarely verbalises concrete plans; criterion (b)/(c)
      is the usual pathway for justified level-3 predictions. Do NOT
      refuse level 3 just because no plan is stated if (b) or (c) holds.

Calibration:
  * Level 1 is the most under-predicted. If the content has passive-only
    death/worthlessness themes, predict 1 (not 0 and not 2).
  * Level 2 is common when ideation is self-directed but no method/plan.
    Do NOT escalate to 3 just because SE/AFF are low.
  * Depression severity alone is NOT sufficient for level \u2265 2. The
    evidence must be ideation/hopelessness-specific.
  * Do NOT collapse to a 0 vs 3 bimodal pattern. Intermediate levels are
    the majority of real cases.

In <analysis>, explicitly state:
  (a) Passive ideation present?  Cite the phrase/indicator.
  (b) Active self-directed-harm ideation present?  Cite.
  (c) Does evidence meet criterion (a)/(b)/(c) of level 3?  Cite.
Pick the HIGHEST level whose evidence is supported.

Answer format (EXACT):
<answer>Suicide=<0-3></answer>"""


#
# Each task is tuned independently; we track the version that achieved the
# highest macro-F1 on the 30-persona subset and lock it here.  Override per
# run via ``BatchDiagnostician(prompt_versions={"suicide": "v4"})``.
# ---------------------------------------------------------------------------
DIAG_DEFAULT_VERSIONS: Dict[str, str] = {
    "big_five": "v6",  # v6 anchors on canonical description + anti-collapse
    # rules for O and N (v5 had bf_macro_f1=0.172 on
    # CRAG n=15 with full O-collapse to 3 and N over-pred).
    "mbti": "v3",  # v3 adds JP-specific anchors and per-axis evidence
    # (v2 had type_acc=0.067 on CRAG n=15; J/P=0.53).
    "depression": "v3",  # macro-F1 0.247-0.271
    "suicide": "v5",  # macro-F1 0.232, broke V3's 0/3 bimodal collapse
}


def get_diag_instruction(task: str, version: Optional[str] = None) -> str:
    """Resolve a diagnosis instruction prompt for ``task`` at the requested
    ``version``.  If ``version`` is None, falls back to
    :data:`DIAG_DEFAULT_VERSIONS`.
    """
    if task not in DIAG_PROMPTS_BY_VERSION:
        raise KeyError(f"Unknown diagnosis task: {task!r}")
    ver = version or DIAG_DEFAULT_VERSIONS[task]
    try:
        return DIAG_PROMPTS_BY_VERSION[task][ver]
    except KeyError as e:
        available = list(DIAG_PROMPTS_BY_VERSION[task].keys())
        raise KeyError(
            f"No prompt version {ver!r} for task {task!r}; available: {available}"
        ) from e


# Backward-compatible flat dict of "current best" prompts.  Used by legacy
# callers; new code should prefer :func:`get_diag_instruction`.
DIAG_INSTRUCTIONS: Dict[str, str] = {
    task: get_diag_instruction(task) for task in DIAG_PROMPTS_BY_VERSION
}


DIAG_USER_TMPL = """{instructions}

=== Structured Interpretation ===
{interpretation_block}

Reason step-by-step, then emit the answer."""


DIAG_USER_TMPL_WITH_PERSONA = """{instructions}

=== Character Description (PRIMARY evidence — anchor to this) ===
{persona_summary}

=== Structured Interpretation (SECONDARY evidence from projective tests) ===
{interpretation_block}

The character description above is the PRIMARY source of truth for personality
traits (Big Five / MBTI). The projective interpretation is a secondary signal
that can REFINE but NOT OVERRIDE the explicit character description.  If the
projective test looks inconsistent with the character's canonical personality
(e.g. a famously stoic character scoring high on projective anxiety), trust
the character description and note the projective artefact.

Reason step-by-step, then emit the answer."""


# --------------------------------------------------------------------------
# Persona-summary loader: extracts the "Role + Attributes" block of a
# CharacterRAG prompt file and truncates to ~1500 chars.  Used by Big
# Five / MBTI diagnosis to anchor predictions on the canonical character
# description instead of relying solely on the projective interpretation
# (which systematically over-predicts neuroticism for fictional characters).
# --------------------------------------------------------------------------

_PERSONA_SUMMARY_CACHE: Dict[str, str] = {}


def load_persona_summary(
    source_type: str,
    source_key: str,
    characters_root: Optional[Path] = None,
    max_chars: int = 1500,
) -> Optional[str]:
    """Return a compact character description for CharacterRAG personas.

    Strips the big "## Knowledge:" dump and keeps only Role + Attributes
    (Belief/Values, Demographics, Psychological Traits, Social Relationships).

    Returns None if the summary cannot be built (AnnaAgent personas, file
    missing, etc.).
    """
    if source_type != "characterrag":
        return None
    cache_key = f"{source_type}::{source_key}"
    if cache_key in _PERSONA_SUMMARY_CACHE:
        return _PERSONA_SUMMARY_CACHE[cache_key] or None
    root = characters_root or (
        Path(__file__).resolve().parents[2] / "characters" / "CharacterRAG"
    )
    path = root / source_key / f"{source_key}_en.prompt.txt"
    if not path.exists():
        _PERSONA_SUMMARY_CACHE[cache_key] = ""
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        _PERSONA_SUMMARY_CACHE[cache_key] = ""
        return None
    # Keep everything before "## Knowledge:" (Role + Attributes only)
    cut = text.find("## Knowledge:")
    if cut > 0:
        text = text[:cut]
    # Drop the "## Role: You are X. Answer the user's questions as if..." line
    import re as _re

    text = _re.sub(
        r"^## Role:\s*\nYou are[^.\n]*\.\s*Answer[^.\n]*\.\s*\n*",
        "",
        text,
        flags=_re.M,
    )
    text = _re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + " …[truncated]"
    _PERSONA_SUMMARY_CACHE[cache_key] = text
    return text


# ============================================================
# Response parsers
# ============================================================


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_answer_block(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    block = _extract_answer_block(text)
    # Try block first, then fallback to first JSON-looking span in full text
    for candidate in (block, text):
        m = _JSON_RE.search(candidate)
        if not m:
            continue
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    return None


def parse_scors_g(text: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    scores: Dict[str, int] = {d: 4 for d in SCORS_G_DIMENSIONS}
    explanations: Dict[str, str] = {}
    data = _extract_json(text) or {}
    for dim in SCORS_G_DIMENSIONS:
        v = data.get(dim)
        if isinstance(v, (int, float)):
            scores[dim] = max(1, min(7, int(v)))
        elif isinstance(v, dict) and "score" in v:
            try:
                scores[dim] = max(1, min(7, int(v["score"])))
            except Exception:
                pass

    # Per-dimension explanations from <analysis>
    m = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
    if m:
        body = m.group(1)
        for dim in SCORS_G_DIMENSIONS:
            em = re.search(rf"(?im)^\s*{dim}\s*[:：]\s*(.+)$", body)
            if em:
                explanations[dim] = em.group(1).strip()[:400]
    return scores, explanations


def parse_sras_encoding(text: str) -> Dict[str, int]:
    data = _extract_json(text) or {}
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[k] = int(v)
        except Exception:
            continue
    return out


def parse_sct_scores(text: str, stem_ids: Sequence[str]) -> Dict[str, int]:
    data = _extract_json(text) or {}
    out: Dict[str, int] = {}
    for sid in stem_ids:
        v = data.get(sid)
        if isinstance(v, (int, float)):
            out[sid] = max(0, min(6, int(v)))
        else:
            out[sid] = 3
    return out


def parse_big_five(text: str) -> Tuple[BigFivePrediction, str]:
    body = _extract_answer_block(text)
    dims = {}
    for d in "OCEAN":
        m = re.search(rf"{d}\s*=\s*(\d)", body)
        dims[d] = int(m.group(1)) if m else 3
        dims[d] = max(1, min(5, dims[d]))
    # thinking / analysis (everything before <answer>)
    ana = text.split("<answer>")[0].strip()
    return (
        BigFivePrediction(
            openness=dims["O"],
            conscientiousness=dims["C"],
            extraversion=dims["E"],
            agreeableness=dims["A"],
            neuroticism=dims["N"],
            explanations={"analysis": ana[:1000]},
        ),
        ana,
    )


def parse_mbti(text: str) -> Tuple[MBTIPrediction, str]:
    body = _extract_answer_block(text).upper()
    m = re.search(r"[EI][SN][TF][JP]", body) or re.search(
        r"[EI][SN][TF][JP]", text.upper()
    )
    code = m.group(0) if m else "XXXX"
    ei = 1.0 if code[0] == "I" else 0.0
    sn = 1.0 if code[1] == "N" else 0.0
    tf = 1.0 if code[2] == "F" else 0.0
    jp = 1.0 if code[3] == "P" else 0.0
    ana = text.split("<answer>")[0].strip()
    pred = MBTIPrediction(
        ei_score=ei,
        sn_score=sn,
        tf_score=tf,
        jp_score=jp,
        explanations={"analysis": ana[:1000]},
    )
    return pred, ana


def parse_depression(text: str) -> Tuple[DepressionPrediction, str]:
    body = _extract_answer_block(text)
    m = re.search(r"Depression\s*=\s*(\d)", body, re.IGNORECASE) or re.search(
        r"Depression\s*=\s*(\d)", text, re.IGNORECASE
    )
    level = int(m.group(1)) if m else 0
    level = max(0, min(3, level))
    ana = text.split("<answer>")[0].strip()
    return DepressionPrediction(level=level, confidence=0.7, explanation=ana[:500]), ana


def parse_suicide(text: str) -> Tuple[SuicidePrediction, str]:
    body = _extract_answer_block(text)
    m = re.search(r"Suicide\s*=\s*(\d)", body, re.IGNORECASE) or re.search(
        r"Suicide\s*=\s*(\d)", text, re.IGNORECASE
    )
    level = int(m.group(1)) if m else 0
    level = max(0, min(3, level))
    ana = text.split("<answer>")[0].strip()
    return SuicidePrediction(level=level, confidence=0.7, explanation=ana[:500]), ana


DIAG_PARSERS = {
    "big_five": parse_big_five,
    "mbti": parse_mbti,
    "depression": parse_depression,
    "suicide": parse_suicide,
}


# ============================================================
# Batch interpreter
# ============================================================


def _format_interpretation_block(result: InterpretationResult) -> str:
    """Render an :class:`InterpretationResult` as a compact text block that
    the diagnostician can read.  Includes numeric scores and key rationales."""
    lines: List[str] = []
    # TAT aggregated
    agg = result.get_aggregated_tat_scores()
    lines.append("TAT / SCORS-G (aggregated across cards, 1–7 scale):")
    for dim in SCORS_G_DIMENSIONS:
        lines.append(f"  {dim}: {agg.get(dim, 4.0):.2f}")
    # Per-card brief
    if result.tat_scores:
        lines.append("\nTAT per-card highlights:")
        for ts in result.tat_scores:
            compact = ", ".join(f"{k}={v}" for k, v in ts.scores.items())
            lines.append(f"  [{ts.response_id}] {compact}")
            # Include most extreme dimensions' explanations
            if ts.explanations:
                extremes = sorted(
                    ts.scores.items(),
                    key=lambda kv: abs(kv[1] - 4),
                    reverse=True,
                )[:2]
                for dim, _ in extremes:
                    expl = ts.explanations.get(dim, "").strip()
                    if expl:
                        lines.append(f"    {dim}: {expl[:180]}")

    # Rorschach
    sras = result.rorschach_scores
    lines.append("\nRorschach / SRAS domain scores:")
    lines.append(f"  CPS (cognitive processing): {sras.cps:.2f}")
    lines.append(f"  ARS (affective regulation): {sras.ars:.2f}")
    lines.append(f"  IRS (interpersonal):        {sras.irs:.2f}")
    lines.append(f"  SCS (stress coping D):      {sras.scs:.2f}")
    if sras.encoding:
        nonzero = {k: v for k, v in sras.encoding.items() if v}
        if nonzero:
            lines.append(
                "  Encoding counts: "
                + ", ".join(f"{k}={v}" for k, v in nonzero.items())
            )

    # SCT
    lines.append("\nSCT domain scores (0=adjusted → 6=conflicted):")
    for d in ["FA", "CA", "SA", "IR", "ER"]:
        v = result.sct_scores.domain_scores.get(d, 3.0)
        lines.append(f"  {d}: {v:.2f}")
    # Highlight most conflicted items
    high_items = sorted(
        result.sct_scores.item_scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:5]
    if high_items:
        lines.append("\nMost conflicted SCT items:")
        for sid, score in high_items:
            expl = result.sct_scores.explanations.get(sid, "").strip()
            lines.append(
                f"  [{sid}] score={score}" + (f" — {expl[:140]}" if expl else "")
            )

    return "\n".join(lines)


@dataclass
class _InterpJob:
    """Book-keeping for a single persona during batched interpretation."""

    persona_key: str
    tat_responses: List[TATResponse]
    rorschach_responses: List[RorschachResponse]
    sct_responses: List[SCTResponse]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchInterpreter:
    """Run Stage 2 for many personas in a single vLLM batch.

    Typical usage::

        llm = VLLMTextClient(model_name=..., enable_thinking=True, enable_lora=...)
        bi  = BatchInterpreter(llm)
        results = bi.interpret_many(jobs)  # dict persona_key -> InterpretationResult
    """

    def __init__(
        self,
        llm: VLLMTextClient,
        config: Optional[GenerationConfig] = None,
        adapter: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.config = config or GenerationConfig(
            max_tokens=3072, temperature=0.3, top_p=0.9
        )
        self.adapter = adapter

    # ---- prompt builders ----

    @staticmethod
    def _scors_prompts(job: _InterpJob) -> List[Tuple[str, str, Dict[str, Any]]]:
        out = []
        for tat in job.tat_responses:
            user = SCORS_G_USER_TMPL.format(
                image_id=tat.image_id,
                narrative=tat.narrative.strip()[:4000],
            )
            out.append(
                (
                    SCORS_G_SYSTEM,
                    user,
                    {
                        "persona": job.persona_key,
                        "kind": "scors",
                        "image_id": tat.image_id,
                    },
                )
            )
        return out

    @staticmethod
    def _sras_prompt(job: _InterpJob) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        if not job.rorschach_responses:
            return None
        cards = []
        for r in job.rorschach_responses:
            inq = r.inquiry or ""
            cards.append(
                f"Card {r.card_number}:\n  Perception: {r.perception.strip()[:800]}"
                + (f"\n  Inquiry:    {inq.strip()[:500]}" if inq else "")
            )
        user = SRAS_USER_TMPL.format(cards_block="\n\n".join(cards))
        return SRAS_SYSTEM, user, {"persona": job.persona_key, "kind": "sras"}

    @staticmethod
    def _sct_prompt(job: _InterpJob) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        if not job.sct_responses:
            return None
        items = [
            f"[{r.stem_id}] Stem: {r.stem.strip()}\n           Completion: {r.completion.strip()[:300]}"
            for r in job.sct_responses
        ]
        user = SCT_USER_TMPL.format(items_block="\n".join(items))
        stems = [r.stem_id for r in job.sct_responses]
        return (
            SCT_SYSTEM,
            user,
            {"persona": job.persona_key, "kind": "sct", "stem_ids": stems},
        )

    # ---- main entry ----

    def interpret_many(
        self,
        jobs: Sequence[_InterpJob],
    ) -> Dict[str, InterpretationResult]:
        """Run interpretation on many personas using a single batched call."""
        all_prompts: List[Tuple[str, str, Dict[str, Any]]] = []
        for job in jobs:
            all_prompts.extend(self._scors_prompts(job))
            sras = self._sras_prompt(job)
            if sras:
                all_prompts.append(sras)
            sct = self._sct_prompt(job)
            if sct:
                all_prompts.append(sct)

        if not all_prompts:
            return {}

        logger.info(
            "BatchInterpreter: dispatching %d prompts to vLLM", len(all_prompts)
        )
        system_prompts = [p[0] for p in all_prompts]
        user_prompts = [p[1] for p in all_prompts]

        # Use chat_batch; each prompt has its own system prompt so expand manually.
        batches: List[List[Dict]] = [
            [{"role": "system", "content": s}, {"role": "user", "content": u}]
            for s, u in zip(system_prompts, user_prompts)
        ]
        outputs = self.llm.generate_batch(
            batches,
            self.config,
            adapter=self.adapter,
            strip_think=True,
        )

        # Regroup outputs by persona
        per_persona: Dict[str, Dict[str, Any]] = {
            job.persona_key: {"scors": [], "sras": None, "sct": None} for job in jobs
        }
        for (_, _, meta), text in zip(all_prompts, outputs):
            pkey = meta["persona"]
            kind = meta["kind"]
            if kind == "scors":
                scores, expls = parse_scors_g(text)
                per_persona[pkey]["scors"].append(
                    SCORSGScore(
                        response_id=meta["image_id"], scores=scores, explanations=expls
                    )
                )
            elif kind == "sras":
                per_persona[pkey]["sras"] = parse_sras_encoding(text)
            elif kind == "sct":
                per_persona[pkey]["sct"] = (
                    parse_sct_scores(text, meta["stem_ids"]),
                    self._sct_extract_expls(text, meta["stem_ids"]),
                )

        # Convert to InterpretationResult
        results: Dict[str, InterpretationResult] = {}
        for job in jobs:
            entry = per_persona[job.persona_key]
            sras_enc = entry["sras"] or {}
            sras = SRASScore(
                encoding=sras_enc,
                cps=self._cps(sras_enc),
                ars=self._ars(sras_enc),
                irs=self._irs(sras_enc),
                scs=self._scs(sras_enc),
                explanations=self._sras_explanations(sras_enc),
            )
            sct_pair = entry["sct"] or ({}, {})
            sct_scores, sct_expls = sct_pair
            sct = SCTScore(
                domain_scores=self._sct_domain_scores(sct_scores),
                item_scores=sct_scores,
                explanations=sct_expls,
            )
            results[job.persona_key] = InterpretationResult(
                tat_scores=entry["scors"],
                rorschach_scores=sras,
                sct_scores=sct,
            )
        return results

    # ---- helpers ported from legacy Interpreter ----

    @staticmethod
    def _cps(e: Dict[str, int]) -> float:
        return (
            2 * e.get("P", 0)
            + e.get("FQo", 0)
            - (e.get("FQu", 0) + 3 * e.get("FQ-", 0) + e.get("WSumCog", 0))
        )

    @staticmethod
    def _ars(e: Dict[str, int]) -> float:
        return 2 * e.get("FC", 0) - (
            e.get("CF", 0)
            + 2 * e.get("C", 0)
            + e.get("C'", 0)
            + e.get("Y", 0)
            + e.get("V", 0)
        )

    @staticmethod
    def _irs(e: Dict[str, int]) -> float:
        return (
            3 * e.get("M", 0)
            + 2 * e.get("COP", 0)
            + e.get("H", 0)
            - (
                2 * (e.get("AGC", 0) + e.get("AGM", 0))
                + 2 * e.get("MOR", 0)
                + 3 * e.get("M-", 0)
            )
        )

    @staticmethod
    def _scs(e: Dict[str, int]) -> float:
        ea = e.get("M", 0) + 0.5 * e.get("FC", 0) + e.get("CF", 0) + 1.5 * e.get("C", 0)
        es = (
            e.get("FM", 0)
            + e.get("m", 0)
            + e.get("Y", 0)
            + e.get("T", 0)
            + e.get("V", 0)
            + e.get("C'", 0)
        )
        return ea - es

    @staticmethod
    def _sras_explanations(enc: Dict[str, int]) -> Dict[str, str]:
        nonzero = {k: v for k, v in enc.items() if v}
        if not nonzero:
            return {}
        return {"encoding": ", ".join(f"{k}={v}" for k, v in nonzero.items())}

    @staticmethod
    def _sct_domain_scores(item_scores: Dict[str, int]) -> Dict[str, float]:
        domain_totals: Dict[str, List[int]] = {
            d: [] for d in ["FA", "CA", "SA", "IR", "ER"]
        }
        for sid, score in item_scores.items():
            sc = sid.rsplit("_", 1)[0]
            domain = SCT_SUBCONSTRUCT_TO_DOMAIN.get(sc)
            if domain:
                domain_totals[domain].append(score)
        return {d: (sum(v) / len(v) if v else 3.0) for d, v in domain_totals.items()}

    @staticmethod
    def _sct_extract_expls(text: str, stem_ids: Sequence[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        m = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
        if not m:
            return out
        body = m.group(1)
        for sid in stem_ids:
            em = re.search(rf"\[\s*{re.escape(sid)}\s*\]\s*(.+)", body)
            if em:
                out[sid] = em.group(1).strip()[:300]
        return out


# ============================================================
# Batch diagnostician
# ============================================================


@dataclass
class DiagJob:
    """A diagnosis request: one persona × one task."""

    persona_key: str
    task: str  # big_five | mbti | depression | suicide
    interpretation: InterpretationResult
    metadata: Dict[str, Any] = field(default_factory=dict)
    persona_summary: Optional[str] = None  # canonical character description
    # (anchors BF/MBTI predictions)


@dataclass
class DiagOutput:
    persona_key: str
    task: str
    prediction: Any  # Big Five / MBTI / Depression / Suicide prediction
    raw_text: str
    analysis: str
    interp_block: str


class BatchDiagnostician:
    """Diagnoses many (persona, task) pairs in one vLLM batch."""

    def __init__(
        self,
        llm: VLLMTextClient,
        config: Optional[GenerationConfig] = None,
        adapter_map: Optional[Dict[str, str]] = None,
        prompt_versions: Optional[Dict[str, str]] = None,
    ) -> None:
        self.llm = llm
        self.config = config or GenerationConfig(
            max_tokens=2048, temperature=0.6, top_p=0.9
        )
        self.adapter_map = adapter_map or {}
        # Resolve per-task prompt versions lazily via get_diag_instruction.
        # ``prompt_versions`` may partially override DIAG_DEFAULT_VERSIONS.
        self.prompt_versions: Dict[str, str] = dict(DIAG_DEFAULT_VERSIONS)
        if prompt_versions:
            self.prompt_versions.update(prompt_versions)

    def diagnose_many(
        self,
        jobs: Sequence[DiagJob],
        *,
        temperature: Optional[float] = None,
        return_thinking: bool = True,
    ) -> List[DiagOutput]:
        if not jobs:
            return []

        cfg = self.config
        if temperature is not None:
            cfg = GenerationConfig(
                max_tokens=cfg.max_tokens,
                temperature=temperature,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                stop_sequences=list(cfg.stop_sequences) if cfg.stop_sequences else None,
            )

        # Group jobs by adapter so we can swap LoRA per group
        per_adapter: Dict[Optional[str], List[int]] = {}
        for i, j in enumerate(jobs):
            adapter = self.adapter_map.get(j.task)
            per_adapter.setdefault(adapter, []).append(i)

        results: List[Optional[DiagOutput]] = [None] * len(jobs)

        for adapter, indices in per_adapter.items():
            batches = []
            interp_blocks = []
            for i in indices:
                j = jobs[i]
                block = _format_interpretation_block(j.interpretation)
                instructions = get_diag_instruction(
                    j.task, self.prompt_versions.get(j.task)
                )
                # For Big Five / MBTI on CharacterRAG personas, inject the
                # canonical character description as PRIMARY evidence.
                # For Depression / Suicide (AnnaAgent) the projective block
                # remains the only signal.
                if j.persona_summary and j.task in ("big_five", "mbti"):
                    user = DIAG_USER_TMPL_WITH_PERSONA.format(
                        instructions=instructions,
                        persona_summary=j.persona_summary,
                        interpretation_block=block,
                    )
                else:
                    user = DIAG_USER_TMPL.format(
                        instructions=instructions,
                        interpretation_block=block,
                    )
                batches.append(
                    [
                        {"role": "system", "content": DIAG_SYSTEM_BASE},
                        {"role": "user", "content": user},
                    ]
                )
                interp_blocks.append(block)

            logger.info(
                "BatchDiagnostician: %d prompts via adapter=%s",
                len(batches),
                adapter,
            )
            outputs = self.llm.generate_batch(
                batches,
                cfg,
                adapter=adapter,
                strip_think=True,
                return_thinking=False,
            )

            for i, text, block in zip(indices, outputs, interp_blocks):
                j = jobs[i]
                parser = DIAG_PARSERS[j.task]
                pred, analysis = parser(text)
                results[i] = DiagOutput(
                    persona_key=j.persona_key,
                    task=j.task,
                    prediction=pred,
                    raw_text=text,
                    analysis=analysis,
                    interp_block=block,
                )

        return [r for r in results if r is not None]


# ============================================================
# Convenience loaders (for scripts)
# ============================================================


def behaviors_to_job(persona_key: str, behavior_dict: Dict) -> _InterpJob:
    """Convert a serialized behavior group into a :class:`_InterpJob`."""
    tat = [TATResponse(**r) for r in behavior_dict.get("tat", [])]
    ror = [RorschachResponse(**r) for r in behavior_dict.get("rorschach", [])]
    sct = [SCTResponse(**r) for r in behavior_dict.get("sct", [])]
    return _InterpJob(
        persona_key=persona_key,
        tat_responses=tat,
        rorschach_responses=ror,
        sct_responses=sct,
        metadata=behavior_dict.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Interpreter teacher helpers (Phase D1 cold-start SFT)
# ---------------------------------------------------------------------------


def format_behaviors_for_interp_teacher(
    behavior_dict: Dict,
    tat_max_chars: int = 1200,
    ror_max_chars: int = 600,
    sct_max_chars: int = 220,
) -> Tuple[str, str, str]:
    """Render a behaviour group into the three text blocks expected by the
    interpreter teacher prompt.  Truncates long narratives to keep total
    prompt within vLLM context (target ≤ 24k tokens input)."""
    tat_items = []
    for r in behavior_dict.get("tat", []):
        nar = r.get("narrative", "").strip()
        if len(nar) > tat_max_chars:
            nar = nar[:tat_max_chars] + " …[truncated]"
        tat_items.append(f"[{r.get('image_id', '?')}] {nar}")
    tat_block = "\n\n".join(tat_items) if tat_items else "(no TAT responses)"

    ror_items = []
    for r in behavior_dict.get("rorschach", []):
        perc = r.get("perception", "").strip()
        inq = (r.get("inquiry") or "").strip()
        if len(perc) > ror_max_chars:
            perc = perc[:ror_max_chars] + " …"
        if len(inq) > ror_max_chars:
            inq = inq[:ror_max_chars] + " …"
        block = f"Card {r.get('card_number', '?')}:\n  Perception: {perc}"
        if inq:
            block += f"\n  Inquiry: {inq}"
        ror_items.append(block)
    rorschach_block = (
        "\n\n".join(ror_items) if ror_items else "(no Rorschach responses)"
    )

    sct_items = []
    for r in behavior_dict.get("sct", []):
        comp = r.get("completion", "").strip()
        if len(comp) > sct_max_chars:
            comp = comp[:sct_max_chars] + " …"
        sct_items.append(
            f"[{r.get('stem_id', '?')}] {r.get('stem', '').strip()} — {comp}"
        )
    sct_block = "\n".join(sct_items) if sct_items else "(no SCT responses)"

    return tat_block, rorschach_block, sct_block


def format_labels_for_interp_teacher(metadata: Dict) -> str:
    """Render available ground-truth labels into a plain-text block for
    insertion into the interpreter teacher user prompt."""
    gt = metadata.get("ground_truth", {}) or {}
    parts = []
    if gt.get("big_five"):
        bf = gt["big_five"]
        parts.append(
            f"Big Five (1=very low, 5=very high): "
            f"O={bf.get('O', '?')} C={bf.get('C', '?')} E={bf.get('E', '?')} "
            f"A={bf.get('A', '?')} N={bf.get('N', '?')}"
        )
    if gt.get("mbti"):
        parts.append(f"MBTI type: {gt['mbti']}")
    if gt.get("depression_level") is not None:
        parts.append(f"Depression level (0=none..3=severe): {gt['depression_level']}")
    if gt.get("suicide_risk") is not None:
        parts.append(f"Suicide risk level (0=none..3=severe): {gt['suicide_risk']}")
    if not parts:
        return "(no ground-truth labels available — score based on behaviours alone)"
    return "\n".join(parts)


def build_interp_teacher_messages(behavior_dict: Dict) -> List[Dict[str, str]]:
    """Return [system, user] messages for the Phase-D1 interpreter teacher."""
    tat_block, ror_block, sct_block = format_behaviors_for_interp_teacher(behavior_dict)
    labels_block = format_labels_for_interp_teacher(behavior_dict.get("metadata", {}))
    user = INTERP_TEACHER_USER_TMPL.format(
        tat_block=tat_block,
        rorschach_block=ror_block,
        sct_block=sct_block,
        labels_block=labels_block,
    )
    return [
        {"role": "system", "content": INTERP_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_interp_production_user(behavior_dict: Dict) -> str:
    """Return the *production* user prompt (labels stripped) — used as the
    user message when emitting the SFT record."""
    tat_block, ror_block, sct_block = format_behaviors_for_interp_teacher(behavior_dict)
    return INTERP_PRODUCTION_USER_TMPL.format(
        tat_block=tat_block,
        rorschach_block=ror_block,
        sct_block=sct_block,
    )


def parse_interp_teacher_output(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON block from a teacher response.  Returns the parsed
    dict on success; None on malformed output.  Accepts optional <think>
    wrapper and trailing whitespace."""
    import re as _re, json as _json

    # Drop any <think>...</think> prelude.
    m = _re.search(r"</think>", raw, flags=_re.DOTALL)
    tail = raw[m.end() :] if m else raw
    # Grab the first balanced {...} block.
    start = tail.find("{")
    if start < 0:
        return None
    depth = 0
    end = None
    for i in range(start, len(tail)):
        ch = tail[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    snippet = tail[start:end]
    try:
        data = _json.loads(snippet)
    except Exception:
        return None
    # Minimal schema check: require at least one of the three top-level keys.
    if not any(k in data for k in ("scors_g", "sras", "sct")):
        return None
    return data


__all__ = [
    "BatchInterpreter",
    "BatchDiagnostician",
    "DiagJob",
    "DiagOutput",
    "_InterpJob",
    "behaviors_to_job",
    "DIAG_INSTRUCTIONS",
    "DIAG_PROMPTS_BY_VERSION",
    "DIAG_DEFAULT_VERSIONS",
    "get_diag_instruction",
    "DIAG_SYSTEM_BASE",
    "DIAG_PARSERS",
    "_format_interpretation_block",
    "parse_big_five",
    "parse_mbti",
    # Phase D1 interpreter teacher helpers
    "INTERP_TEACHER_SYSTEM",
    "INTERP_TEACHER_USER_TMPL",
    "INTERP_PRODUCTION_USER_TMPL",
    "format_behaviors_for_interp_teacher",
    "format_labels_for_interp_teacher",
    "build_interp_teacher_messages",
    "build_interp_production_user",
    "parse_interp_teacher_output",
    "parse_depression",
    "parse_suicide",
]
