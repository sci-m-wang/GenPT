#!/usr/bin/env python3
"""
Evaluate questionnaire-based psychological assessment as a baseline.

This implements the "traditional" questionnaire approach for comparison:
  1. Have the LLM role-play as a character (same as GenPT examinee)
  2. Present standardized questionnaire items
  3. Score responses using questionnaire scoring rules
  4. Compare with the same ground truth labels used by GenPT

Questionnaires used:
  - BFI (Big Five Inventory, 44 items)  → Big Five personality levels (1-5)
  - 16Personalities (60 items)           → MBTI type
  - BDI (Beck's Depression Inventory, 21 items) → Depression risk (0-5)
  - BSS (Beck Scale for Suicide Ideation, 19 items) → Suicide risk (0-4)

Usage:
    # Run full questionnaire evaluation
    python scripts/evaluate_questionnaire.py --gpu 0

    # Run only CharacterRAG (Big Five + MBTI)
    python scripts/evaluate_questionnaire.py --source characterrag --gpu 0

    # Run only AnnaAgent (Depression + Suicide)
    python scripts/evaluate_questionnaire.py --source annaagent --gpu 0

    # Compare with GenPT results after evaluation
    python scripts/evaluate_questionnaire.py --compare-only
"""

import argparse
import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_questionnaire")

# ============================================================
# Scoring mappings
# ============================================================

# BDI sum (0-63) → depression risk level (0-3, matching AnnaAgent portrait.drisk)
BDI_THRESHOLDS = [
    (13, 0),   # 0-13:  minimal
    (19, 1),   # 14-19: mild
    (28, 2),   # 20-28: moderate
    (63, 3),   # 29-63: severe
]

# BSS sum (0-38) → suicide risk level (0-3, matching AnnaAgent portrait.srisk)
BSS_THRESHOLDS = [
    (3, 0),    # 0-3:   none
    (8, 1),    # 4-8:   low
    (16, 2),   # 9-16:  moderate
    (38, 3),   # 17-38: high
]

BF_LONG_TO_SHORT = {
    'openness': 'O', 'conscientiousness': 'C', 'extraversion': 'E',
    'agreeableness': 'A', 'neuroticism': 'N',
}


def map_score_to_level(score: int, thresholds: list) -> int:
    for upper, level in thresholds:
        if score <= upper:
            return level
    return thresholds[-1][1]


# ============================================================
# Questionnaire loading
# ============================================================

def load_questionnaire(name: str) -> dict:
    path = ROOT / "questionnaires" / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Character loading
# ============================================================

def load_characterrag_characters() -> List[Dict[str, Any]]:
    """Load CharacterRAG characters with prompts and ground truth."""
    pdb_path = ROOT / "characters" / "CharacterRAG" / "pdb_labels.json"
    with open(pdb_path) as f:
        pdb = json.load(f)

    chars = []
    char_dir = ROOT / "characters" / "CharacterRAG"
    for char_key, gt_data in pdb.items():
        prompt_file = char_dir / char_key / f"{char_key}_en.prompt.txt"
        if not prompt_file.exists():
            logger.warning(f"Prompt file not found: {prompt_file}")
            continue
        system_prompt = prompt_file.read_text(encoding="utf-8")

        # Extract Big Five ground truth levels
        gt_bf = {}
        for long_name, short in BF_LONG_TO_SHORT.items():
            entry = gt_data.get('big_five', {}).get(long_name, {})
            gt_bf[short] = entry.get('level', 3) if isinstance(entry, dict) else 3

        # Extract MBTI ground truth
        mbti_data = gt_data.get('mbti', {})
        gt_mbti = mbti_data.get('type', '????') if isinstance(mbti_data, dict) else str(mbti_data)

        chars.append({
            "name": char_key,
            "system_prompt": system_prompt,
            "gt_bf": gt_bf,
            "gt_mbti": gt_mbti,
        })

    return chars


def load_annaagent_characters(
    match_genpt_round: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load AnnaAgent characters with prompts and ground truth.

    Args:
        match_genpt_round: If set, only load characters that have GenPT
            evaluation results from this round (for fair comparison).
    """
    labels_path = ROOT / "characters" / "AnnaAgent" / "D4_prompts_with_labels.json"
    with open(labels_path, encoding="utf-8") as f:
        all_data = json.load(f)

    # Optionally filter to match GenPT evaluation set
    allowed_ids: Optional[set] = None
    if match_genpt_round is not None:
        results_dir = ROOT / "results" / f"round_{match_genpt_round}" / "annaagent"
        if results_dir.exists():
            allowed_ids = set()
            for fp in results_dir.glob("*.json"):
                if any(kw in fp.name for kw in ['error', 'summary', 'stats']):
                    continue
                try:
                    with open(fp) as f:
                        result = json.load(f)
                    sid = result.get('source_id', '')
                    if sid:
                        allowed_ids.add(sid)
                except Exception:
                    pass
            logger.info(f"Filtering to {len(allowed_ids)} AnnaAgent IDs from round {match_genpt_round}")

    # Build ID lookup for all labels
    id_to_entry: Dict[str, dict] = {}
    for entry in all_data:
        eid = entry.get('id', '')
        id_to_entry[eid] = entry

    chars = []
    if allowed_ids is not None:
        # Match by exact ID or prefix
        for sid in allowed_ids:
            entry = id_to_entry.get(sid)
            if entry is None:
                # Try prefix matching
                for k, v in id_to_entry.items():
                    if sid and k and (sid.startswith(k[:8]) or k.startswith(sid[:8])):
                        entry = v
                        break
            if entry is None:
                continue
            label = entry.get('label', {})
            chars.append({
                "name": entry.get('id', '')[:16],
                "source_id": entry.get('id', ''),
                "system_prompt": entry.get('prompt', ''),
                "gt_depression": label.get('drisk', 0),
                "gt_suicide": label.get('srisk', 0),
            })
    else:
        for entry in all_data:
            eid = entry.get('id', '')
            label = entry.get('label', {})
            chars.append({
                "name": eid[:16],
                "source_id": eid,
                "system_prompt": entry.get('prompt', ''),
                "gt_depression": label.get('drisk', 0),
                "gt_suicide": label.get('srisk', 0),
            })

    return chars


# ============================================================
# Prompt builders
# ============================================================

def build_bfi_prompt(questionnaire: dict) -> str:
    """Build prompt presenting all 44 BFI items at once."""
    questions = questionnaire["questions"]
    lines = [
        "I am going to present you with a series of statements about personality.",
        'Each statement begins with "I see myself as someone who..."',
        "For each statement, please indicate how much you agree or disagree.",
        "Use a scale from 1 to 5:",
        "  1 = Strongly disagree",
        "  2 = A little disagree",
        "  3 = Neither agree nor disagree",
        "  4 = A little agree",
        "  5 = Strongly agree",
        "",
        "Reply ONLY with numbered scores, one per line, like:",
        "1: 3",
        "2: 5",
        "...",
        "",
        "Here are the statements (\"I see myself as someone who...\"):",
    ]
    for qid in sorted(questions.keys(), key=int):
        q = questions[qid]
        lines.append(f"{qid}. {q['origin_en']}")

    return "\n".join(lines)


def build_16p_prompt(questionnaire: dict) -> str:
    """Build prompt presenting all 60 16Personalities items at once.

    NOTE: The original 16P uses 1=agree, 7=disagree, but LLMs naturally
    associate higher numbers with more agreement.  We present the standard
    Likert direction (1=disagree, 7=agree) and adjust scoring accordingly.
    """
    questions = questionnaire["questions"]
    lines = [
        "I am going to present you with a series of statements about yourself.",
        "For each statement, please indicate how much you agree or disagree.",
        "Use a scale from 1 to 7:",
        "  1 = Strongly disagree",
        "  2 = Disagree",
        "  3 = Somewhat disagree",
        "  4 = Neither agree nor disagree",
        "  5 = Somewhat agree",
        "  6 = Agree",
        "  7 = Strongly agree",
        "",
        "Reply ONLY with numbered scores, one per line, like:",
        "1: 4",
        "2: 2",
        "...",
        "",
        "Here are the statements:",
    ]
    for qid in sorted(questions.keys(), key=int):
        q = questions[qid]
        lines.append(f"{qid}. {q['origin_en']}")

    return "\n".join(lines)


def build_bdi_prompt(questionnaire: dict) -> str:
    """Build prompt presenting all 21 BDI items with options."""
    questions = questionnaire["questions"]
    lines = [
        "请仔细阅读以下每组陈述，然后选择最能描述你过去两周（包括今天）感受的那一项。",
        "在每组中选择一个数字（0-3），0表示最轻微，3表示最严重。",
        "",
        "请只输出每题的编号和得分，每行一个，例如：",
        "1: 0",
        "2: 2",
        "...",
        "",
        "以下是各组陈述：",
    ]
    for qid in sorted(questions.keys(), key=int):
        q = questions[qid]
        lines.append(f"\n{qid}. {q['origin_zh']}")
        for score_str in sorted(q["options"].keys(), key=int):
            option = q["options"][score_str]
            lines.append(f"  {score_str}: {option['zh']}")

    return "\n".join(lines)


def build_bss_prompt(questionnaire: dict) -> str:
    """Build prompt presenting all 19 BSS items with options."""
    questions = questionnaire["questions"]
    lines = [
        "请根据你最近一周的感受，对以下每个问题选择最符合的选项。",
        "每题选择一个数字（0-2），0表示程度最低，2表示程度最高。",
        "",
        "请只输出每题的编号和得分，每行一个，例如：",
        "1: 0",
        "2: 1",
        "...",
        "",
        "以下是各题：",
    ]
    for qid in sorted(questions.keys(), key=int):
        q = questions[qid]
        lines.append(f"\n{qid}. {q['rewritten_zh']}")
        for score_str in sorted(q["options"].keys(), key=int):
            option = q["options"][score_str]
            lines.append(f"  {score_str}: {option['zh']}")

    return "\n".join(lines)


# ============================================================
# Response parsing
# ============================================================

def parse_numbered_scores(
    text: str, num_questions: int, low: int, high: int,
) -> Dict[int, int]:
    """Parse numbered scores from model output like '1: 3\\n2: 5\\n...'"""
    scores: Dict[int, int] = {}
    # Match patterns: "1: 3", "1:3", "1. 3", "1) 3"
    for match in re.finditer(r'(\d+)\s*[:.)]\s*(\d+)', text):
        qid = int(match.group(1))
        score = int(match.group(2))
        if 1 <= qid <= num_questions and low <= score <= high:
            scores[qid] = score
    return scores


# ============================================================
# Scoring functions
# ============================================================

def score_bfi(
    scores: Dict[int, int], questionnaire: dict,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Score BFI responses → Big Five levels (1-5).

    Returns:
        (levels_dict {O/C/E/A/N: 1-5}, averages_dict {O/C/E/A/N: float})
    """
    reverse_items = set(questionnaire["reverse"])
    scale = questionnaire["scale"]  # 6 → reversed = 6 - score

    # Apply reverse scoring
    adjusted: Dict[int, int] = {}
    for qid, score in scores.items():
        if qid in reverse_items:
            adjusted[qid] = scale - score
        else:
            adjusted[qid] = score

    # Dimension name → short code
    dim_map = {
        'Extraversion': 'E', 'Agreeableness': 'A',
        'Conscientiousness': 'C', 'Neuroticism': 'N', 'Openness': 'O',
    }

    levels: Dict[str, int] = {}
    averages: Dict[str, float] = {}
    for cat in questionnaire["categories"]:
        dim_name = cat["cat_name"]
        short = dim_map.get(dim_name, dim_name[0])
        q_ids = cat["cat_questions"]

        dim_scores = [adjusted[qid] for qid in q_ids if qid in adjusted]
        if dim_scores:
            avg = sum(dim_scores) / len(dim_scores)
            averages[short] = round(avg, 3)
            levels[short] = max(1, min(5, round(avg)))
        else:
            averages[short] = 3.0
            levels[short] = 3

    return levels, averages


def score_16p(scores: Dict[int, int], questionnaire: dict) -> str:
    """Score 16Personalities responses → MBTI type code.

    Scoring logic:
    - Reverse scored items: adjusted = scale - raw  (scale=8)
    - After reverse scoring, all items in a dimension point same direction
    - LOW sum → first pole (E, S, T, P); HIGH sum → second pole (I, N, F, J)
    """
    reverse_items = set(questionnaire["reverse"])
    scale = questionnaire["scale"]  # 8 → reversed = 8 - score

    adjusted: Dict[int, int] = {}
    for qid, score in scores.items():
        if qid in reverse_items:
            adjusted[qid] = scale - score
        else:
            adjusted[qid] = score

    dim_letters = {
        'E/I': ('E', 'I'),
        'S/N': ('S', 'N'),
        'T/F': ('T', 'F'),
        'P/J': ('P', 'J'),
    }

    mbti: List[str] = []
    for cat in questionnaire["categories"]:
        dim_name = cat["cat_name"]
        first, second = dim_letters[dim_name]
        q_ids = cat["cat_questions"]

        dim_scores = [adjusted[qid] for qid in q_ids if qid in adjusted]
        if dim_scores:
            total = sum(dim_scores)
            # Midpoint: each item midpoint=4 (range 1-7), so total midpoint = n*4
            # With our 1=disagree,7=agree prompt: HIGH sum → first pole
            midpoint = len(dim_scores) * 4
            mbti.append(first if total > midpoint else second)
        else:
            mbti.append('?')

    return ''.join(mbti)


def score_bdi(scores: Dict[int, int], questionnaire: dict) -> Tuple[int, int]:
    """Score BDI responses → (total_sum, depression_level 0-5)."""
    reverse_items = set(questionnaire.get("reverse", []))
    total = 0
    for qid, score in scores.items():
        if qid in reverse_items:
            total += (3 - score)
        else:
            total += score
    level = map_score_to_level(total, BDI_THRESHOLDS)
    return total, level


def score_bss(scores: Dict[int, int], questionnaire: dict) -> Tuple[int, int]:
    """Score BSS responses → (total_sum, suicide_level 0-4)."""
    reverse_items = set(questionnaire.get("reverse", []))
    total = 0
    for qid, score in scores.items():
        if qid in reverse_items:
            total += (2 - score)
        else:
            total += score
    level = map_score_to_level(total, BSS_THRESHOLDS)
    return total, level


# ============================================================
# Metrics (identical to evaluate_adapters.py)
# ============================================================

def _macro_f1(preds: list, gts: list, classes: Optional[list] = None) -> float:
    """Compute macro-averaged F1 across all classes."""
    if classes is None:
        classes = sorted(set(preds) | set(gts))
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gts) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compute_metrics(predictions: dict) -> dict:
    """Compute evaluation metrics from collected predictions.

    Metrics:
      - Big Five:    macro F1 (5-class ordinal per dimension) + per-dim F1
      - MBTI:        dimension accuracy + MAE (since it has continuous %)
      - Depression:   macro F1 (4-class ordinal 0-3)
      - Suicide:      macro F1 (4-class ordinal 0-3)
    """
    metrics: Dict[str, Any] = {}

    if "big_five" in predictions and predictions["big_five"]:
        all_preds, all_gts = [], []
        metrics["bf_n"] = len(predictions["big_five"])
        for dim in 'OCEAN':
            dp = [p.get(dim, 3) for p, g in predictions["big_five"]]
            dg = [g.get(dim, 3) for p, g in predictions["big_five"]]
            all_preds.extend(dp)
            all_gts.extend(dg)
            metrics[f"bf_{dim}_f1"] = round(_macro_f1(dp, dg, list(range(1, 6))), 3)
        metrics["bf_macro_f1"] = round(_macro_f1(all_preds, all_gts, list(range(1, 6))), 3)
        # Also keep MAE for reference
        errors = [abs(p - g) for p, g in zip(all_preds, all_gts)]
        metrics["bf_mae"] = round(sum(errors) / len(errors), 3) if errors else None

    if "mbti" in predictions and predictions["mbti"]:
        total_dims = 0
        matches = 0
        type_matches = 0
        for pred_type, gt_type in predictions["mbti"]:
            if len(pred_type) == 4 and len(gt_type) == 4:
                total_dims += 4
                dim_match = sum(1 for a, b in zip(pred_type, gt_type) if a == b)
                matches += dim_match
                if dim_match == 4:
                    type_matches += 1
        metrics["mbti_dim_accuracy"] = round(
            matches / total_dims, 3) if total_dims else None
        metrics["mbti_type_accuracy"] = round(
            type_matches / len(predictions["mbti"]), 3) if predictions["mbti"] else None
        metrics["mbti_n"] = len(predictions["mbti"])

    if "depression" in predictions and predictions["depression"]:
        dp = [p for p, g in predictions["depression"]]
        dg = [g for p, g in predictions["depression"]]
        metrics["dep_macro_f1"] = round(_macro_f1(dp, dg, list(range(4))), 3)
        metrics["dep_n"] = len(predictions["depression"])
        exact = sum(1 for p, g in zip(dp, dg) if p == g)
        metrics["dep_exact"] = round(exact / len(dp), 3)
        # Also keep MAE for reference
        errors = [abs(p - g) for p, g in zip(dp, dg)]
        metrics["dep_mae"] = round(sum(errors) / len(errors), 3)

    if "suicide" in predictions and predictions["suicide"]:
        sp = [p for p, g in predictions["suicide"]]
        sg = [g for p, g in predictions["suicide"]]
        metrics["sui_macro_f1"] = round(_macro_f1(sp, sg, list(range(4))), 3)
        metrics["sui_n"] = len(predictions["suicide"])
        exact = sum(1 for p, g in zip(sp, sg) if p == g)
        metrics["sui_exact"] = round(exact / len(sp), 3)
        # Also keep MAE for reference
        errors = [abs(p - g) for p, g in zip(sp, sg)]
        metrics["sui_mae"] = round(sum(errors) / len(errors), 3)

    return metrics


# ============================================================
# Administer questionnaire via LLM
# ============================================================

def administer_questionnaire(client, system_prompt: str, user_prompt: str, config) -> str:
    """Have the LLM answer a questionnaire in character."""
    from genpt.llm.base import Message
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    return client.generate(messages, config)


# ============================================================
# Main evaluation
# ============================================================

def run_evaluation(
    sources: List[str],
    gpu_id: int = 0,
    match_genpt_round: int = 1,
) -> dict:
    """Run the full questionnaire-based evaluation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from genpt.config import EXAMINEE_MODEL_CONFIG
    from genpt.llm.qwen import QwenVLClient
    from genpt.llm.base import GenerationConfig

    model_path = EXAMINEE_MODEL_CONFIG.model_path or EXAMINEE_MODEL_CONFIG.model_name
    logger.info(f"Loading VL model (examinee): {model_path} on GPU {gpu_id}")

    client = QwenVLClient(
        model_name=model_path,
        use_api=False,
        enable_thinking=False,  # questionnaire answering does not need thinking
    )
    client._load_local_model()

    # Generation config: low temperature for consistent answers, enough tokens
    config = GenerationConfig(
        max_tokens=4096,
        temperature=0.3,
        top_p=0.9,
    )

    # Load questionnaires
    bfi_q = load_questionnaire("BFI")
    p16_q = load_questionnaire("16Personalities")
    bdi_q = load_questionnaire("BDI")
    bss_q = load_questionnaire("BSS")

    # Pre-build prompts
    bfi_prompt = build_bfi_prompt(bfi_q)
    p16_prompt = build_16p_prompt(p16_q)
    bdi_prompt = build_bdi_prompt(bdi_q)
    bss_prompt = build_bss_prompt(bss_q)

    predictions: Dict[str, list] = {
        "big_five": [], "mbti": [], "depression": [], "suicide": [],
    }
    raw_outputs: List[dict] = []
    detail_records: List[dict] = []  # per-character detailed results

    # === CharacterRAG (Big Five + MBTI) ===
    if "characterrag" in sources:
        chars = load_characterrag_characters()
        logger.info(f"=== CharacterRAG: {len(chars)} characters ===")

        for i, char in enumerate(chars):
            logger.info(f"[{i+1}/{len(chars)}] {char['name']}")
            record = {"name": char["name"], "source": "characterrag"}

            # --- BFI → Big Five ---
            start = time.time()
            bfi_output = administer_questionnaire(
                client, char["system_prompt"], bfi_prompt, config)
            elapsed = time.time() - start
            bfi_scores = parse_numbered_scores(bfi_output, 44, 1, 5)

            if len(bfi_scores) >= 30:
                bf_levels, bf_avgs = score_bfi(bfi_scores, bfi_q)
                predictions["big_five"].append((bf_levels, char["gt_bf"]))
                record["bfi"] = {
                    "parsed": len(bfi_scores), "total": 44,
                    "averages": bf_avgs, "levels": bf_levels,
                    "gt": char["gt_bf"], "elapsed": round(elapsed, 1),
                }
                logger.info(
                    f"  BFI: {len(bfi_scores)}/44 parsed | "
                    f"pred={bf_levels} gt={char['gt_bf']} ({elapsed:.1f}s)")
            else:
                record["bfi"] = {"parsed": len(bfi_scores), "total": 44, "error": "too few"}
                logger.warning(f"  BFI: only {len(bfi_scores)}/44 parsed, skipping")

            raw_outputs.append({
                "name": char["name"], "source": "characterrag", "task": "big_five",
                "raw_output": bfi_output, "parsed_count": len(bfi_scores),
                "elapsed": round(elapsed, 1),
            })

            # --- 16Personalities → MBTI ---
            start = time.time()
            p16_output = administer_questionnaire(
                client, char["system_prompt"], p16_prompt, config)
            elapsed = time.time() - start
            p16_scores = parse_numbered_scores(p16_output, 60, 1, 7)

            if len(p16_scores) >= 40:
                mbti_pred = score_16p(p16_scores, p16_q)
                predictions["mbti"].append((mbti_pred, char["gt_mbti"]))
                record["16p"] = {
                    "parsed": len(p16_scores), "total": 60,
                    "mbti_pred": mbti_pred, "gt": char["gt_mbti"],
                    "elapsed": round(elapsed, 1),
                }
                logger.info(
                    f"  16P: {len(p16_scores)}/60 parsed | "
                    f"pred={mbti_pred} gt={char['gt_mbti']} ({elapsed:.1f}s)")
            else:
                record["16p"] = {"parsed": len(p16_scores), "total": 60, "error": "too few"}
                logger.warning(f"  16P: only {len(p16_scores)}/60 parsed, skipping")

            raw_outputs.append({
                "name": char["name"], "source": "characterrag", "task": "mbti",
                "raw_output": p16_output, "parsed_count": len(p16_scores),
                "elapsed": round(elapsed, 1),
            })

            detail_records.append(record)

    # === AnnaAgent (Depression + Suicide) ===
    if "annaagent" in sources:
        chars = load_annaagent_characters(match_genpt_round=match_genpt_round)
        logger.info(f"=== AnnaAgent: {len(chars)} characters ===")

        for i, char in enumerate(chars):
            logger.info(f"[{i+1}/{len(chars)}] {char['name']}")
            record = {
                "name": char["name"], "source_id": char["source_id"],
                "source": "annaagent",
            }

            # --- BDI → Depression ---
            start = time.time()
            bdi_output = administer_questionnaire(
                client, char["system_prompt"], bdi_prompt, config)
            elapsed = time.time() - start
            bdi_scores = parse_numbered_scores(bdi_output, 21, 0, 3)

            if len(bdi_scores) >= 15:
                bdi_total, dep_level = score_bdi(bdi_scores, bdi_q)
                predictions["depression"].append((dep_level, char["gt_depression"]))
                record["bdi"] = {
                    "parsed": len(bdi_scores), "total": 21,
                    "sum": bdi_total, "level": dep_level,
                    "gt": char["gt_depression"], "elapsed": round(elapsed, 1),
                }
                logger.info(
                    f"  BDI: {len(bdi_scores)}/21 parsed | "
                    f"sum={bdi_total} level={dep_level} gt={char['gt_depression']} ({elapsed:.1f}s)")
            else:
                record["bdi"] = {"parsed": len(bdi_scores), "total": 21, "error": "too few"}
                logger.warning(f"  BDI: only {len(bdi_scores)}/21 parsed, skipping")

            raw_outputs.append({
                "name": char["name"], "source": "annaagent", "task": "depression",
                "raw_output": bdi_output, "parsed_count": len(bdi_scores),
                "elapsed": round(elapsed, 1),
            })

            # --- BSS → Suicide ---
            start = time.time()
            bss_output = administer_questionnaire(
                client, char["system_prompt"], bss_prompt, config)
            elapsed = time.time() - start
            bss_scores = parse_numbered_scores(bss_output, 19, 0, 2)

            if len(bss_scores) >= 13:
                bss_total, sui_level = score_bss(bss_scores, bss_q)
                predictions["suicide"].append((sui_level, char["gt_suicide"]))
                record["bss"] = {
                    "parsed": len(bss_scores), "total": 19,
                    "sum": bss_total, "level": sui_level,
                    "gt": char["gt_suicide"], "elapsed": round(elapsed, 1),
                }
                logger.info(
                    f"  BSS: {len(bss_scores)}/19 parsed | "
                    f"sum={bss_total} level={sui_level} gt={char['gt_suicide']} ({elapsed:.1f}s)")
            else:
                record["bss"] = {"parsed": len(bss_scores), "total": 19, "error": "too few"}
                logger.warning(f"  BSS: only {len(bss_scores)}/19 parsed, skipping")

            raw_outputs.append({
                "name": char["name"], "source": "annaagent", "task": "suicide",
                "raw_output": bss_output, "parsed_count": len(bss_scores),
                "elapsed": round(elapsed, 1),
            })

            detail_records.append(record)

    # Compute metrics
    metrics = compute_metrics(predictions)
    metrics["method"] = "questionnaire"
    metrics["timestamp"] = datetime.now().isoformat()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("QUESTIONNAIRE EVALUATION RESULTS")
    logger.info("=" * 60)
    if metrics.get("bf_macro_f1") is not None:
        logger.info(f"Big Five macro-F1: {metrics['bf_macro_f1']:.3f}  MAE={metrics.get('bf_mae', '?')}  (n={metrics['bf_n']})")
        for dim in 'OCEAN':
            logger.info(f"  {dim} F1: {metrics[f'bf_{dim}_f1']:.3f}")
    if metrics.get("mbti_dim_accuracy") is not None:
        logger.info(f"MBTI dim accuracy: {metrics['mbti_dim_accuracy']:.3f}  (n={metrics['mbti_n']})")
        logger.info(f"MBTI type exact:   {metrics.get('mbti_type_accuracy', 0):.3f}")
    if metrics.get("dep_macro_f1") is not None:
        logger.info(f"Depression F1:     {metrics['dep_macro_f1']:.3f}  MAE={metrics.get('dep_mae', '?')}  (n={metrics['dep_n']}) "
                     f"exact={metrics['dep_exact']:.3f}")
    if metrics.get("sui_macro_f1") is not None:
        logger.info(f"Suicide F1:        {metrics['sui_macro_f1']:.3f}  MAE={metrics.get('sui_mae', '?')}  (n={metrics['sui_n']}) "
                     f"exact={metrics['sui_exact']:.3f}")
    logger.info("=" * 60)

    # Save results
    eval_dir = ROOT / "results" / "questionnaire"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(eval_dir / f"metrics_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(eval_dir / f"raw_outputs_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, indent=2, ensure_ascii=False)
    with open(eval_dir / f"details_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(detail_records, f, indent=2, ensure_ascii=False)

    # Save predictions for later comparison
    pred_serializable: Dict[str, list] = {}
    for task, pairs in predictions.items():
        pred_serializable[task] = [
            (dict(p) if isinstance(p, dict) else p,
             dict(g) if isinstance(g, dict) else g)
            for p, g in pairs
        ]
    with open(eval_dir / f"predictions_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(pred_serializable, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to: {eval_dir}")
    return metrics


# ============================================================
# Comparison with GenPT results
# ============================================================

def find_latest_metrics(directory: Path, prefix: str = "metrics_") -> Optional[dict]:
    """Find and load the latest metrics JSON in a directory."""
    if not directory.exists():
        return None
    files = sorted(directory.glob(f"{prefix}*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def compare_results():
    """Compare questionnaire results with GenPT projective test results."""
    results_base = ROOT / "results"

    # Load questionnaire metrics
    q_metrics = find_latest_metrics(results_base / "questionnaire")

    # Load GenPT metrics (baseline, R3, R4)
    genpt_baseline = None
    genpt_r3 = None
    genpt_r4 = None

    for r in [4, 3, 2, 1]:
        eval_dir = results_base / f"round_{r}" / "eval"
        baseline = find_latest_metrics(eval_dir, "metrics_baseline")
        adapter = find_latest_metrics(eval_dir, "metrics_adapter")
        if baseline and genpt_baseline is None:
            genpt_baseline = baseline
        if adapter:
            if r == 3 and genpt_r3 is None:
                genpt_r3 = adapter
            elif r == 4 and genpt_r4 is None:
                genpt_r4 = adapter

    # Print comparison table
    print("\n" + "=" * 80)
    print("方法对比: 量表法 vs GenPT投射测试")
    print("=" * 80)

    def fmt(v, lower_better=True):
        if v is None:
            return "  —  "
        return f"{v:.3f}"

    rows = [
        ("Big Five F1 ↑", "bf_macro_f1"),
        ("  O F1", "bf_O_f1"),
        ("  C F1", "bf_C_f1"),
        ("  E F1", "bf_E_f1"),
        ("  A F1", "bf_A_f1"),
        ("  N F1", "bf_N_f1"),
        ("Big Five MAE ↓", "bf_mae"),
        ("MBTI dim acc ↑", "mbti_dim_accuracy"),
        ("MBTI type acc ↑", "mbti_type_accuracy"),
        ("Depression F1 ↑", "dep_macro_f1"),
        ("Depression exact ↑", "dep_exact"),
        ("Suicide F1 ↑", "sui_macro_f1"),
        ("Suicide exact ↑", "sui_exact"),
    ]

    # Header
    print(f"{'指标':<20} {'量表法':>10} {'GenPT-Base':>12} {'GenPT-R3':>10} {'GenPT-R4':>10}")
    print("-" * 65)

    for label, key in rows:
        q_val = q_metrics.get(key) if q_metrics else None
        b_val = genpt_baseline.get(key) if genpt_baseline else None
        r3_val = genpt_r3.get(key) if genpt_r3 else None
        r4_val = genpt_r4.get(key) if genpt_r4 else None

        print(f"{label:<20} {fmt(q_val):>10} {fmt(b_val):>12} {fmt(r3_val):>10} {fmt(r4_val):>10}")

    print("=" * 65)
    print("↓ = lower is better, ↑ = higher is better")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Questionnaire-based evaluation baseline")
    parser.add_argument(
        "--source",
        choices=["characterrag", "annaagent", "all"],
        default="all",
        help="Which data source to evaluate",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--match-round", type=int, default=1,
        help="Only evaluate AnnaAgent chars that have GenPT results from this round",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only print comparison (skip evaluation)",
    )
    args = parser.parse_args()

    if args.compare_only:
        compare_results()
        return

    sources = (
        ["characterrag", "annaagent"]
        if args.source == "all"
        else [args.source]
    )

    run_evaluation(sources=sources, gpu_id=args.gpu, match_genpt_round=args.match_round)

    # Auto-show comparison after evaluation
    compare_results()


if __name__ == "__main__":
    main()
