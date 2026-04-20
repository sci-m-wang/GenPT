#!/usr/bin/env python3
"""
Generate fixed behavior sets for all characters.

This script runs Stage 1 (Examinee) to produce N_GROUPS sets of
behavioral responses (TAT, Rorschach, SCT) per character.

These fixed responses are later analyzed multiple times by Stage 2+3
(multi-round analysis sampling) to generate diverse training data.

Usage:
    # Start vLLM server first:
    vllm serve /path/to/Qwen3-VL-8B-Instruct --tensor-parallel-size 2

    # Generate 5 behavior groups per character
    python scripts/generate_behaviors.py --num-groups 5 --output-dir data/behaviors

    # Generate for specific source only
    python scripts/generate_behaviors.py --sources characterrag --num-groups 3
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Allow running from scripts/ or project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from genpt.config import (
    EXAMINEE_MODEL_CONFIG,
    TAT_DIR,
    RORSCHACH_DIR,
    SCT_DATA_FILE,
    CHARACTERRAG_DIR,
    ANNAAGENT_DIR,
)
from genpt.llm.qwen import create_client_from_config
from genpt.llm.base import GenerationConfig
from genpt.stimuli.tat import TATStimuli
from genpt.stimuli.rorschach import RorschachStimuli
from genpt.stimuli.sct import SCTStimuli
from genpt.pipeline.examinee import (
    Examinee, CharacterLoader,
    TATResponse, RorschachResponse, SCTResponse,
)
from genpt.pipeline.validity import build_condition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_behaviors")


def _load_pdb_labels():
    """Load ground truth labels from pdb_labels.json."""
    labels_path = CHARACTERRAG_DIR / "pdb_labels.json"
    if not labels_path.exists():
        logger.warning("pdb_labels.json not found at %s", labels_path)
        return {}
    with open(labels_path, encoding="utf-8") as f:
        raw = json.load(f)
    # Convert to simplified format: char_name -> {big_five: {O,C,E,A,N}, mbti: str}
    labels = {}
    dim_map = {
        "openness": "O", "conscientiousness": "C", "extraversion": "E",
        "agreeableness": "A", "neuroticism": "N",
    }
    for name, data in raw.items():
        bf = {}
        for dim_name, dim_data in data.get("big_five", {}).items():
            short = dim_map.get(dim_name)
            if short:
                bf[short] = dim_data["level"]
        mbti = data.get("mbti", {}).get("type")
        labels[name] = {"big_five": bf if len(bf) == 5 else None, "mbti": mbti}
    return labels


def load_characters(sources, annaagent_selected_path=None):
    """
    Load all character personas.

    Returns list of (persona, source_type, source_key) tuples.
    """
    characters = []

    if "characterrag" in sources:
        pdb_labels = _load_pdb_labels()
        prompt_files = sorted(CHARACTERRAG_DIR.glob("*/*_en.prompt.txt"))
        for pf in prompt_files:
            char_name = pf.parent.name
            persona = CharacterLoader.load_from_characterrag_prompt(pf, char_name)
            # Inject ground truth from pdb_labels.json
            gt = pdb_labels.get(char_name, {})
            if gt.get("big_five"):
                persona.big_five = gt["big_five"]
            if gt.get("mbti"):
                persona.mbti = gt["mbti"]
            characters.append((persona, "characterrag", char_name))
        logger.info("Loaded %d CharacterRAG characters (with GT labels)", len(prompt_files))

    if "annaagent" in sources:
        # Use the curated 30 selection if available
        selected_path = annaagent_selected_path or (
            ANNAAGENT_DIR / "selected_30.json"
        )
        if selected_path.exists():
            with open(selected_path, encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                personas = CharacterLoader.load_from_annaagent(
                    ANNAAGENT_DIR / "D4_prompts_with_labels.json",
                    character_id=item["id"],
                )
                if personas:
                    characters.append((personas[0], "annaagent", item["id"]))
            logger.info("Loaded %d AnnaAgent characters from selection", len(items))
        else:
            logger.warning("No selected_30.json found; loading first 30 from full set")
            all_personas = CharacterLoader.load_from_annaagent(
                ANNAAGENT_DIR / "D4_prompts_with_labels.json"
            )
            for p in all_personas[:30]:
                characters.append((p, "annaagent", p.source_id))

    return characters


def generate_one_group(
    examinee: Examinee,
    tat_images,
    rorschach_cards,
    sct_stems,
    include_inquiry: bool = True,
):
    """Run full Stage 1 assessment and return serializable dict."""
    tat_responses = []
    for img in tat_images:
        resp = examinee.respond_tat(img)
        tat_responses.append({
            "image_id": resp.image_id,
            "narrative": resp.narrative,
        })

    rorschach_raw = examinee.respond_rorschach(
        rorschach_cards, include_inquiry=include_inquiry
    )
    rorschach_responses = [
        {
            "card_number": r.card_number,
            "perception": r.perception,
            "inquiry": r.inquiry,
        }
        for r in rorschach_raw
    ]

    sct_responses = []
    for stem in sct_stems:
        resp = examinee.respond_sct(stem)
        sct_responses.append({
            "stem_id": resp.stem_id,
            "stem": resp.stem,
            "completion": resp.completion,
        })

    return {
        "tat": tat_responses,
        "rorschach": rorschach_responses,
        "sct": sct_responses,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed behavior sets (Stage 1) for all characters"
    )
    parser.add_argument(
        "--num-groups", type=int, default=5,
        help="Number of behavior groups to generate per character (default: 5)",
    )
    parser.add_argument(
        "--sources", nargs="+", default=["characterrag", "annaagent"],
        choices=["characterrag", "annaagent"],
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/behaviors",
        help="Output directory (default: data/behaviors)",
    )
    parser.add_argument(
        "--annaagent-selection", type=str, default=None,
        help="Path to AnnaAgent selected JSON (default: characters/AnnaAgent/selected_30.json)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Generation temperature (default: 0.8)",
    )
    parser.add_argument(
        "--num-tat", type=int, default=8,
        help="Number of TAT images per assessment (default: 8)",
    )
    parser.add_argument(
        "--num-sct", type=int, default=20,
        help="Number of SCT stems per assessment (default: 20)",
    )
    parser.add_argument(
        "--stimuli-seed", type=int, default=42,
        help="Random seed for stimuli selection (default: 42)",
    )
    parser.add_argument(
        "--rotate-stimuli", action="store_true",
        help="Use a different stimuli subset per group by seeding with "
             "`stimuli_seed + group_idx`.  group_0 still uses the base seed "
             "(backward compatible with existing data/behaviors/).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip character-group combos that already have output files",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU device ID (for local model)",
    )
    parser.add_argument(
        "--shard-idx", type=int, default=0,
        help="Shard index for multi-GPU parallelism (default: 0)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards for multi-GPU parallelism (default: 1)",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="Override EXAMINEE_MODEL_CONFIG.api_base (e.g. http://localhost:8000/v1). "
             "When set, the VL model is served via a pre-launched vLLM OpenAI server "
             "instead of loading via local transformers.",
    )
    parser.add_argument(
        "--condition", type=str, default="baseline",
        choices=["baseline", "sdb_job", "sdb_clinical", "longctx"],
        help="Validity-test perturbation injected at examinee stage. "
             "baseline = no perturbation (re-creates data/behaviors_subset). "
             "sdb_job / sdb_clinical = extra system message describing the "
             "social situation (job interview vs. clinical intake), inserted "
             "between persona and elicitation. "
             "longctx = 10 prior turns of real conversation (AnnaAgent D4_S) "
             "or 10 CRAG Q&A pairs injected as multi-turn history before the "
             "elicitation turn.",
    )
    parser.add_argument(
        "--longctx-turns", type=int, default=10,
        help="Number of prior turns for --condition longctx (default: 10).",
    )
    args = parser.parse_args()

    # ── Apply api_base override (vLLM server mode) ──
    if args.api_base:
        EXAMINEE_MODEL_CONFIG.api_base = args.api_base
        # In server mode, use the served-model-name ("Qwen/Qwen3-VL-8B-Instruct"),
        # not the local filesystem path which vLLM does not register.
        EXAMINEE_MODEL_CONFIG.model_path = None
        logger.info("Using vLLM server at %s for examinee model", args.api_base)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load stimuli pools (selection done per-group below) ──
    logger.info("Loading stimuli pools...")
    tat_stimuli = TATStimuli(stimuli_dir=TAT_DIR)
    rorschach_stimuli = RorschachStimuli(stimuli_dir=RORSCHACH_DIR)
    sct_stimuli = SCTStimuli(data_file=SCT_DATA_FILE)

    def select_stimuli_for_group(group_idx: int):
        """Return (tat_images, rorschach_cards, sct_stems) for group_idx.

        When --rotate-stimuli is set, seed = stimuli_seed + group_idx so
        each group samples a different subset from the pool.  group_0
        always uses the bare stimuli_seed so legacy data/behaviors/ output
        stays byte-compatible."""
        if args.rotate_stimuli and group_idx > 0:
            seed = args.stimuli_seed + group_idx
        else:
            seed = args.stimuli_seed
        tat = tat_stimuli.select_for_assessment(num_total=args.num_tat, seed=seed)
        ror = rorschach_stimuli.select_variants(seed=seed)[:10]
        sct = sct_stimuli.select_for_assessment(num_total=args.num_sct, seed=seed)
        return tat, ror, sct

    # Log stimuli summary at group 0
    _tat, _ror, _sct = select_stimuli_for_group(0)
    logger.info(
        "Stimuli per group: %d TAT, %d Rorschach, %d SCT  (rotate=%s)",
        len(_tat), len(_ror), len(_sct), args.rotate_stimuli,
    )
    del _tat, _ror, _sct

    # ── Initialize VL model ──
    logger.info("Initializing examinee model: %s", EXAMINEE_MODEL_CONFIG.model_name)
    vl_model = create_client_from_config(EXAMINEE_MODEL_CONFIG)

    gen_config = GenerationConfig(
        max_tokens=1024,
        temperature=args.temperature,
    )

    # ── Load characters ──
    anna_sel = Path(args.annaagent_selection) if args.annaagent_selection else None
    characters = load_characters(args.sources, anna_sel)
    logger.info("Loaded %d characters total", len(characters))

    # ── Apply sharding ──
    if args.num_shards > 1:
        characters = characters[args.shard_idx::args.num_shards]
        logger.info(
            "Shard %d/%d: processing %d characters",
            args.shard_idx, args.num_shards, len(characters),
        )

    # ── Generate ──
    total = len(characters) * args.num_groups
    done = 0
    start_time = datetime.now()

    for persona, source_type, source_key in characters:
        safe_key = source_key.replace("/", "_").replace(" ", "_")
        char_dir = output_dir / source_type / safe_key
        char_dir.mkdir(parents=True, exist_ok=True)

        # Build per-persona validity-test perturbation (if any).
        extra_sys, prefix_hist = build_condition(
            args.condition,
            PROJECT_ROOT,
            source_type,
            source_key,
            num_turns=args.longctx_turns,
        )

        examinee = Examinee(
            vl_model,
            persona,
            gen_config,
            extra_system_messages=extra_sys,
            prefix_history=prefix_hist,
        )

        for g in range(args.num_groups):
            out_path = char_dir / f"group_{g}.json"
            if args.skip_existing and out_path.exists():
                logger.info("Skip existing: %s group %d", source_key, g)
                done += 1
                continue

            logger.info(
                "[%d/%d] %s / %s / group %d",
                done + 1, total, source_type, source_key, g,
            )
            try:
                tat_images, rorschach_cards, sct_stems = select_stimuli_for_group(g)
                behaviors = generate_one_group(
                    examinee, tat_images, rorschach_cards, sct_stems
                )
                behaviors["metadata"] = {
                    "source_type": source_type,
                    "source_key": source_key,
                    "group": g,
                    "persona_name": persona.name,
                    "temperature": args.temperature,
                    "timestamp": datetime.now().isoformat(),
                    "condition": args.condition,
                    "longctx_turns": (args.longctx_turns if args.condition == "longctx" else 0),
                    "stimuli_seed": (args.stimuli_seed + g) if (args.rotate_stimuli and g > 0) else args.stimuli_seed,
                    "stimuli_ids": {
                        "tat": [getattr(i, "id", None) or getattr(i, "image_id", None) or str(i) for i in tat_images],
                        "rorschach": [getattr(c, "id", None) or getattr(c, "card_id", None) or str(c) for c in rorschach_cards],
                        "sct": [getattr(s, "id", None) or getattr(s, "stem_id", None) or str(s) for s in sct_stems],
                    },
                    "ground_truth": {
                        "big_five": persona.big_five,
                        "mbti": persona.mbti,
                        "depression_level": persona.depression_level,
                        "suicide_risk": persona.suicide_risk,
                    },
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(behaviors, f, ensure_ascii=False, indent=2)
                logger.info("  Saved: %s", out_path)
            except Exception as e:
                logger.error("  Failed: %s group %d: %s", source_key, g, e)

            done += 1

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("Done. %d/%d groups generated in %.1fs", done, total, elapsed)


if __name__ == "__main__":
    main()
