#!/usr/bin/env python
"""
GenPT Experiment Runner — Round 1 (Initial Inference)

Runs the full GenPT assessment pipeline on:
1. CharacterRAG: 15 fictional characters (personality tasks)
2. AnnaAgent: subset of profiles (mental health tasks)

All raw responses and results are saved to results/.
"""

import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genpt.config import (
    CHARACTERRAG_DIR,
    ANNAAGENT_DATA_PATH,
    PDB_LABELS_PATH,
    EXAMINEE_MODEL_CONFIG,
    INTERPRETER_MODEL_CONFIG,
    DEFAULT_STIMULI_CONFIG,
)
from genpt.llm.qwen import create_client_from_config
from genpt.experiments.run_assessment import (
    AssessmentConfig,
    AssessmentResult,
    run_single_assessment,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("genpt.experiment")

RESULTS_BASE = Path("results")
ROUND_DIR = RESULTS_BASE / "round_1"
CHARACTERRAG_RESULTS = ROUND_DIR / "characterrag"
ANNAAGENT_RESULTS = ROUND_DIR / "annaagent"


def get_characterrag_characters() -> List[Dict]:
    """List all CharacterRAG characters with profile paths and ground truth."""
    with open(PDB_LABELS_PATH, "r", encoding="utf-8") as f:
        pdb_labels = json.load(f)

    characters = []
    for char_name, labels in pdb_labels.items():
        char_dir = CHARACTERRAG_DIR / char_name
        # Prefer English profile
        profile_path = char_dir / f"{char_name}_en.txt"
        if not profile_path.exists():
            profile_path = char_dir / f"{char_name}.txt"
        if not profile_path.exists():
            logger.warning("No profile for %s, skipping", char_name)
            continue

        gt = {
            "mbti_type": labels.get("mbti", {}).get("type"),
            "big_five": {
                "O": labels.get("big_five", {}).get("openness", {}).get("level", 3),
                "C": labels.get("big_five", {})
                .get("conscientiousness", {})
                .get("level", 3),
                "E": labels.get("big_five", {}).get("extraversion", {}).get("level", 3),
                "A": labels.get("big_five", {})
                .get("agreeableness", {})
                .get("level", 3),
                "N": labels.get("big_five", {}).get("neuroticism", {}).get("level", 3),
            },
        }
        characters.append(
            {"name": char_name, "profile_path": str(profile_path), "ground_truth": gt}
        )
    return characters


def get_annaagent_profiles(max_count: int = 50) -> List[Dict]:
    """Load AnnaAgent profiles (up to max_count)."""
    with open(ANNAAGENT_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = []
    for entry in data[:max_count]:
        gt = {
            "depression_level": entry.get("label", {}).get("drisk", 0),
            "suicide_level": entry.get("label", {}).get("srisk", 0),
        }
        profiles.append(
            {
                "id": entry["id"],
                "prompt": entry.get("prompt", ""),
                "ground_truth": gt,
            }
        )
    return profiles


def run_characterrag_experiments(
    examinee_model, interpreter_model, resume_from: Optional[str] = None
) -> List[AssessmentResult]:
    """Run assessments on all CharacterRAG characters."""
    CHARACTERRAG_RESULTS.mkdir(parents=True, exist_ok=True)
    characters = get_characterrag_characters()
    logger.info("CharacterRAG: %d characters found", len(characters))

    results = []
    skip = resume_from is not None
    for i, char in enumerate(characters):
        if skip:
            if char["name"] == resume_from:
                skip = False
            else:
                logger.info("Skipping %s (resuming from %s)", char["name"], resume_from)
                continue

        logger.info("=" * 60)
        logger.info("[%d/%d] CharacterRAG: %s", i + 1, len(characters), char["name"])
        logger.info("=" * 60)

        # Check if already done (handle case differences: saved as Anya_Forger, name is anya_forger)
        existing = [
            f
            for f in CHARACTERRAG_RESULTS.glob("*.json")
            if f.stem.lower().startswith(char["name"].lower())
            and "ground_truth" not in f.stem
            and "error" not in f.stem
        ]
        if existing:
            logger.info("Already completed, loading from %s", existing[0])
            with open(existing[0]) as f:
                results.append(json.load(f))
            continue

        config = AssessmentConfig(
            source_type="characterrag",
            character_path=char["profile_path"],
            num_tat_images=DEFAULT_STIMULI_CONFIG.num_tat_images,
            num_rorschach_cards=DEFAULT_STIMULI_CONFIG.num_rorschach_cards,
            num_sct_stems=DEFAULT_STIMULI_CONFIG.num_sct_stems,
            output_dir=str(CHARACTERRAG_RESULTS),
        )

        start = time.time()
        try:
            result = run_single_assessment(
                config,
                examinee_model=examinee_model,
                interpreter_model=interpreter_model,
                diagnostician_model=interpreter_model,
            )
            elapsed = time.time() - start
            logger.info("Completed %s in %.1fs", char["name"], elapsed)

            # Save ground truth alongside result
            gt_path = CHARACTERRAG_RESULTS / f"{char['name']}_ground_truth.json"
            with open(gt_path, "w") as f:
                json.dump(char["ground_truth"], f, indent=2)

            results.append(result)
        except Exception as e:
            logger.error(
                "Failed on %s: %s\n%s", char["name"], e, traceback.format_exc()
            )
            # Save error for debugging
            err_path = CHARACTERRAG_RESULTS / f"{char['name']}_error.json"
            with open(err_path, "w") as f:
                json.dump(
                    {
                        "name": char["name"],
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    f,
                    indent=2,
                )

    return results


def run_annaagent_experiments(
    examinee_model,
    interpreter_model,
    max_count: int = 50,
    resume_from: Optional[str] = None,
) -> List[AssessmentResult]:
    """Run assessments on AnnaAgent profiles."""
    ANNAAGENT_RESULTS.mkdir(parents=True, exist_ok=True)
    profiles = get_annaagent_profiles(max_count=max_count)
    logger.info("AnnaAgent: %d profiles to run", len(profiles))

    results = []
    skip = resume_from is not None
    for i, prof in enumerate(profiles):
        pid = prof["id"]
        short_id = pid[:12]

        if skip:
            if pid == resume_from:
                skip = False
            else:
                continue

        logger.info("=" * 60)
        logger.info("[%d/%d] AnnaAgent: %s", i + 1, len(profiles), short_id)
        logger.info("=" * 60)

        # Check if already done
        existing = list(ANNAAGENT_RESULTS.glob(f"{short_id}_*.json"))
        if existing:
            logger.info("Already completed, loading from %s", existing[0])
            with open(existing[0]) as f:
                results.append(json.load(f))
            continue

        config = AssessmentConfig(
            source_type="annaagent",
            character_path=str(ANNAAGENT_DATA_PATH),
            character_id=pid,
            num_tat_images=DEFAULT_STIMULI_CONFIG.num_tat_images,
            num_rorschach_cards=DEFAULT_STIMULI_CONFIG.num_rorschach_cards,
            num_sct_stems=DEFAULT_STIMULI_CONFIG.num_sct_stems,
            output_dir=str(ANNAAGENT_RESULTS),
        )

        start = time.time()
        try:
            result = run_single_assessment(
                config,
                examinee_model=examinee_model,
                interpreter_model=interpreter_model,
                diagnostician_model=interpreter_model,
            )
            elapsed = time.time() - start
            logger.info("Completed %s in %.1fs", short_id, elapsed)

            # Save ground truth
            gt_path = ANNAAGENT_RESULTS / f"{short_id}_ground_truth.json"
            with open(gt_path, "w") as f:
                json.dump(prof["ground_truth"], f, indent=2)

            results.append(result)
        except Exception as e:
            logger.error("Failed on %s: %s\n%s", short_id, e, traceback.format_exc())
            err_path = ANNAAGENT_RESULTS / f"{short_id}_error.json"
            with open(err_path, "w") as f:
                json.dump(
                    {"id": pid, "error": str(e), "traceback": traceback.format_exc()},
                    f,
                    indent=2,
                )

    return results


def summarize_characterrag_results(results_dir: Path):
    """Print summary of CharacterRAG results vs ground truth."""
    gt_files = sorted(results_dir.glob("*_ground_truth.json"))
    if not gt_files:
        logger.info("No results to summarize.")
        return

    print("\n" + "=" * 80)
    print("CHARACTERRAG RESULTS SUMMARY")
    print("=" * 80)

    for gt_file in gt_files:
        char_name = gt_file.stem.replace("_ground_truth", "")
        with open(gt_file) as f:
            gt = json.load(f)

        # Find matching result
        result_files = list(results_dir.glob(f"{char_name}_2*.json"))
        if not result_files:
            print(f"\n{char_name}: NO RESULT")
            continue

        with open(result_files[0]) as f:
            result = json.load(f)

        diag = result.get("diagnosis", {})
        bf_pred = diag.get("big_five", {})
        mbti_pred = diag.get("mbti", {})

        print(f"\n--- {char_name} ---")
        print(
            f"  MBTI:  predicted={mbti_pred.get('type', '?')}, gt={gt.get('mbti_type', '?')}"
        )
        if bf_pred and gt.get("big_five"):
            for trait in ["O", "C", "E", "A", "N"]:
                pred_val = bf_pred.get(trait, "?")
                if isinstance(pred_val, dict):
                    pred_val = pred_val.get("level", pred_val.get("score", "?"))
                gt_val = gt["big_five"].get(trait, "?")
                match = (
                    "✓"
                    if pred_val == gt_val
                    else f"diff={abs(int(pred_val or 0) - int(gt_val or 0))}"
                )
                print(f"  Big5 {trait}: pred={pred_val}, gt={gt_val}  ({match})")


def main():
    """Main experiment flow."""
    import argparse

    parser = argparse.ArgumentParser(description="Run GenPT experiments")
    parser.add_argument(
        "--scope", choices=["characterrag", "annaagent", "all"], default="all"
    )
    parser.add_argument(
        "--annaagent-count",
        type=int,
        default=50,
        help="Number of AnnaAgent profiles to run (default: 50, use 0 for all)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from this character/profile ID",
    )
    args = parser.parse_args()

    # Ensure output dirs
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    # Initialize models once
    logger.info(
        "Initializing Examinee model (VL): %s", EXAMINEE_MODEL_CONFIG.model_path
    )
    examinee_model = create_client_from_config(EXAMINEE_MODEL_CONFIG)

    logger.info(
        "Initializing Interpreter model (text): %s", INTERPRETER_MODEL_CONFIG.model_path
    )
    interpreter_model = create_client_from_config(INTERPRETER_MODEL_CONFIG)

    # Run experiments
    if args.scope in ("characterrag", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING CHARACTERRAG EXPERIMENTS")
        logger.info("=" * 80)
        cr_results = run_characterrag_experiments(
            examinee_model,
            interpreter_model,
            resume_from=args.resume_from if args.scope == "characterrag" else None,
        )
        summarize_characterrag_results(CHARACTERRAG_RESULTS)

    if args.scope in ("annaagent", "all"):
        count = args.annaagent_count or 1338  # 0 means all
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING ANNAAGENT EXPERIMENTS (%d profiles)", count)
        logger.info("=" * 80)
        aa_results = run_annaagent_experiments(
            examinee_model,
            interpreter_model,
            max_count=count,
            resume_from=args.resume_from if args.scope == "annaagent" else None,
        )

    logger.info("All experiments complete. Results saved to %s", ROUND_DIR)


if __name__ == "__main__":
    main()
