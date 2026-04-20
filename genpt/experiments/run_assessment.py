"""
GenPT Assessment Runner

Main entry point for running psychological assessments using the GenPT framework.
Supports both AnnaAgent and CharacterRAG character sources.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict

from ..config import (
    TAT_DIR,
    RORSCHACH_DIR,
    SCT_DATA_FILE,
    SCORS_G_DIMENSIONS,
    SRAS_DOMAIN_SCORES,
    SCT_DOMAINS,
    DEFAULT_STIMULI_CONFIG,
    DEFAULT_MODEL_CONFIG,
    EXAMINEE_MODEL_CONFIG,
    INTERPRETER_MODEL_CONFIG,
)
from ..llm.qwen import QwenVLClient, QwenTextClient, create_client_from_config
from ..llm.base import BaseLLM, GenerationConfig
from ..stimuli.tat import TATStimuli
from ..stimuli.rorschach import RorschachStimuli
from ..stimuli.sct import SCTStimuli
from ..pipeline.examinee import (
    Examinee, Persona, CharacterLoader,
    TATResponse, RorschachResponse, SCTResponse,
)
from ..pipeline.interpreter import Interpreter, InterpretationResult
from ..pipeline.diagnostician import Diagnostician


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("genpt.assessment")


@dataclass
class AssessmentConfig:
    """Configuration for a single assessment run."""
    # Character source
    source_type: str = "characterrag"  # "annaagent" or "characterrag"
    character_path: str = ""
    character_id: Optional[str] = None
    
    # Stimuli selection (paper: 8 TAT 4:3:1, 10 Rorschach, 20 SCT)
    num_tat_images: int = 8
    tat_categories: Optional[List[str]] = None
    num_rorschach_cards: int = 10
    num_sct_stems: int = 20
    sct_domains: Optional[List[str]] = None
    
    # Assessment options
    include_rorschach_inquiry: bool = True
    
    # Model settings
    model_name: str = EXAMINEE_MODEL_CONFIG.model_name
    use_api: bool = True
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    
    # Generation settings
    max_tokens: Optional[int] = None  # None = no limit, model generates until EOS
    temperature: float = 0.8
    
    # Output
    output_dir: str = "./results"
    save_raw_responses: bool = True


@dataclass 
class AssessmentResult:
    """Complete result from an assessment."""
    config: AssessmentConfig
    persona_name: str
    source_type: str
    source_id: str
    
    # Raw responses
    tat_responses: List[Dict]
    rorschach_responses: List[Dict]
    sct_responses: List[Dict]
    
    # Interpreted scores
    interpretation: Optional[Dict] = None
    
    # Final diagnosis
    diagnosis: Optional[Dict] = None
    
    # Metadata
    timestamp: str = ""
    duration_seconds: float = 0.0


def run_single_assessment(
    config: AssessmentConfig,
    examinee_model: Optional[BaseLLM] = None,
    interpreter_model: Optional[BaseLLM] = None,
    diagnostician_model: Optional[BaseLLM] = None,
) -> AssessmentResult:
    """
    Run a complete psychological assessment on a single character.
    
    Args:
        config: Assessment configuration
        examinee_model: Pre-initialized VL LLM for examinee
        interpreter_model: Pre-initialized text LLM for interpreter
        diagnostician_model: Pre-initialized text LLM for diagnostician
        
    Returns:
        AssessmentResult with all responses and analysis
    """
    start_time = datetime.now()
    logger.info(f"Starting assessment with config: {config.source_type}/{config.character_path}")
    
    # Initialize models — separate VL and text models
    if examinee_model is None:
        logger.info("Initializing examinee model (VL): %s", EXAMINEE_MODEL_CONFIG.model_name)
        examinee_model = create_client_from_config(EXAMINEE_MODEL_CONFIG)
    
    if interpreter_model is None:
        logger.info("Initializing interpreter model (text): %s", INTERPRETER_MODEL_CONFIG.model_name)
        interpreter_model = create_client_from_config(INTERPRETER_MODEL_CONFIG)
    
    if diagnostician_model is None:
        diagnostician_model = interpreter_model  # share text model
    
    generation_config = GenerationConfig(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    
    # Load persona
    logger.info("Loading persona...")
    if config.source_type == "annaagent":
        examinee = Examinee.from_annaagent(
            llm=examinee_model,
            json_path=config.character_path,
            character_id=config.character_id,
            generation_config=generation_config,
        )
    elif config.source_type == "characterrag":
        examinee = Examinee.from_characterrag(
            llm=examinee_model,
            profile_path=config.character_path,
            generation_config=generation_config,
        )
    else:
        raise ValueError(f"Unknown source type: {config.source_type}")
    
    persona = examinee.persona
    logger.info(f"Loaded persona: {persona.name}")
    
    # Load stimuli
    logger.info("Loading stimuli...")
    tat_stimuli = TATStimuli(stimuli_dir=TAT_DIR)
    rorschach_stimuli = RorschachStimuli(stimuli_dir=RORSCHACH_DIR)
    sct_stimuli = SCTStimuli(data_file=SCT_DATA_FILE)
    
    # Select TAT images (4:3:1 ratio per paper)
    tat_images = tat_stimuli.select_for_assessment(
        num_total=config.num_tat_images,
        seed=42,
    )
    
    # Select Rorschach cards
    rorschach_cards = rorschach_stimuli.select_variants(seed=42)
    rorschach_cards = rorschach_cards[:config.num_rorschach_cards]
    
    # Select SCT stems (4 per domain)
    sct_stems = sct_stimuli.select_for_assessment(
        num_total=config.num_sct_stems,
        seed=42,
    )
    
    logger.info(f"Selected: {len(tat_images)} TAT, {len(rorschach_cards)} Rorschach, {len(sct_stems)} SCT")
    
    # Stage 1: Generate responses
    logger.info("Stage 1: Generating examinee responses...")
    
    tat_responses = []
    for i, img in enumerate(tat_images):
        logger.info(f"  TAT {i+1}/{len(tat_images)}: {img.id}")
        response = examinee.respond_tat(img)
        tat_responses.append({
            "image_id": response.image_id,
            "narrative": response.narrative,
        })
    
    logger.info(f"  Rorschach ({len(rorschach_cards)} cards)...")
    rorschach_responses_raw = examinee.respond_rorschach(
        rorschach_cards, 
        include_inquiry=config.include_rorschach_inquiry
    )
    rorschach_responses = [
        {
            "card_number": r.card_number,
            "perception": r.perception,
            "inquiry": r.inquiry,
        }
        for r in rorschach_responses_raw
    ]
    
    sct_responses = []
    for i, stem in enumerate(sct_stems):
        logger.info(f"  SCT {i+1}/{len(sct_stems)}: {stem.id}")
        response = examinee.respond_sct(stem)
        sct_responses.append({
            "stem_id": response.stem_id,
            "stem": response.stem,
            "completion": response.completion,
        })
    
    # Stage 2: Interpret responses
    logger.info("Stage 2: Interpreting responses...")
    interpreter = Interpreter(interpreter_model)
    
    # Reconstruct typed response objects for interpreter
    tat_resp_objects = [TATResponse(**r) for r in tat_responses]
    sct_resp_objects = [SCTResponse(**r) for r in sct_responses]

    interpretation_result = interpreter.interpret_all(
        tat_responses=tat_resp_objects,
        rorschach_responses=rorschach_responses_raw,
        sct_responses=sct_resp_objects,
    )
    
    # Serialize InterpretationResult properly
    # tat_scores: List[SCORSGScore] — each has response_id, scores (Dict), explanations (Dict)
    # rorschach_scores: SRASScore — single object with encoding, cps/ars/irs/scs, explanations
    # sct_scores: SCTScore — single object with domain_scores, item_scores, explanations
    interpretation_dict = {
        "tat_scores": [
            {
                "response_id": s.response_id,
                "scores": s.scores,
                "explanations": s.explanations,
                "mean_score": s.mean_score(),
            }
            for s in interpretation_result.tat_scores
        ],
        "rorschach_scores": {
            "encoding": interpretation_result.rorschach_scores.encoding,
            "cps": interpretation_result.rorschach_scores.cps,
            "ars": interpretation_result.rorschach_scores.ars,
            "irs": interpretation_result.rorschach_scores.irs,
            "scs": interpretation_result.rorschach_scores.scs,
            "explanations": interpretation_result.rorschach_scores.explanations,
        },
        "sct_scores": {
            "domain_scores": interpretation_result.sct_scores.domain_scores,
            "item_scores": interpretation_result.sct_scores.item_scores,
            "explanations": interpretation_result.sct_scores.explanations,
        },
        "aggregated_tat_scores": interpretation_result.get_aggregated_tat_scores(),
    }
    
    # Stage 3: Make diagnoses
    logger.info("Stage 3: Making diagnoses...")
    diagnostician = Diagnostician(diagnostician_model)
    
    diagnosis_result = diagnostician.diagnose_all(interpretation_result)
    
    diagnosis_dict = {
        "big_five": {**diagnosis_result.big_five.to_dict(), "explanations": diagnosis_result.big_five.explanations} if diagnosis_result.big_five else None,
        "mbti": asdict(diagnosis_result.mbti) if diagnosis_result.mbti else None,
        "depression": asdict(diagnosis_result.depression) if diagnosis_result.depression else None,
        "suicide": asdict(diagnosis_result.suicide) if diagnosis_result.suicide else None,
    }
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Assessment completed in {duration:.1f}s")
    
    result = AssessmentResult(
        config=config,
        persona_name=persona.name,
        source_type=persona.source_type,
        source_id=persona.source_id or "",
        tat_responses=tat_responses,
        rorschach_responses=rorschach_responses,
        sct_responses=sct_responses,
        interpretation=interpretation_dict,
        diagnosis=diagnosis_dict,
        timestamp=start_time.isoformat(),
        duration_seconds=duration,
    )
    
    # Save results
    if config.output_dir:
        save_results(result, config.output_dir)
    
    return result


def run_batch_assessment(
    configs: List[AssessmentConfig],
    shared_model: bool = True,
) -> List[AssessmentResult]:
    """
    Run assessments on multiple characters.
    
    Args:
        configs: List of assessment configurations
        shared_model: Whether to share a single model across assessments
        
    Returns:
        List of AssessmentResult objects
    """
    results = []
    
    # Initialize shared models if requested
    examinee = None
    interp = None
    if shared_model and configs:
        logger.info("Initializing shared examinee model (VL): %s", EXAMINEE_MODEL_CONFIG.model_name)
        examinee = create_client_from_config(EXAMINEE_MODEL_CONFIG)
        logger.info("Initializing shared interpreter/diagnostician model (text): %s", INTERPRETER_MODEL_CONFIG.model_name)
        interp = create_client_from_config(INTERPRETER_MODEL_CONFIG)
    
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Assessment {i+1}/{len(configs)}")
        logger.info(f"{'='*60}")
        
        try:
            result = run_single_assessment(
                config,
                examinee_model=examinee,
                interpreter_model=interp,
                diagnostician_model=interp,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            raise
    
    return results


def save_results(result: AssessmentResult, output_dir: Union[str, Path]) -> Path:
    """Save assessment results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.persona_name.replace(' ', '_')}_{timestamp}.json"
    output_path = output_dir / filename
    
    # Convert to dict for JSON serialization
    result_dict = {
        "persona_name": result.persona_name,
        "source_type": result.source_type,
        "source_id": result.source_id,
        "timestamp": result.timestamp,
        "duration_seconds": result.duration_seconds,
        "config": asdict(result.config),
        "responses": {
            "tat": result.tat_responses,
            "rorschach": result.rorschach_responses,
            "sct": result.sct_responses,
        },
        "interpretation": result.interpretation,
        "diagnosis": result.diagnosis,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run GenPT psychological assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on CharacterRAG character
  python -m genpt.experiments.run_assessment \\
    --source characterrag \\
    --character-path characters/CharacterRAG/anya_forger/anya_forger_en.txt
  
  # Run on AnnaAgent character
  python -m genpt.experiments.run_assessment \\
    --source annaagent \\
    --character-path characters/AnnaAgent/D4_prompts.json \\
    --character-id test9d8cca9d-4c81-4bb7-96e6-9b0fb6d967fc
  
  # Use API instead of local model
  python -m genpt.experiments.run_assessment \\
    --source characterrag \\
    --character-path characters/CharacterRAG/anya_forger/anya_forger_en.txt \\
    --use-api \\
    --api-key YOUR_API_KEY
        """
    )
    
    # Character source
    parser.add_argument("--source", choices=["characterrag", "annaagent"], 
                        default="characterrag", help="Character data source")
    parser.add_argument("--character-path", required=True, 
                        help="Path to character file/directory")
    parser.add_argument("--character-id", help="Character ID (for AnnaAgent)")
    
    # Stimuli options
    parser.add_argument("--num-tat", type=int, default=8, 
                        help="Number of TAT images (paper default: 8)")
    parser.add_argument("--num-rorschach", type=int, default=10, 
                        help="Number of Rorschach cards")
    parser.add_argument("--num-sct", type=int, default=20, 
                        help="Number of SCT stems")
    parser.add_argument("--skip-inquiry", action="store_true",
                        help="Skip Rorschach inquiry phase")
    
    # Model options
    parser.add_argument("--model", default=EXAMINEE_MODEL_CONFIG.model_name,
                        help="Examinee model name")
    parser.add_argument("--use-api", action="store_true",
                        help="Use API instead of local model")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--api-key", help="API key")
    
    # Generation options
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Generation temperature")
    
    # Output
    parser.add_argument("--output-dir", default="./results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Build config
    config = AssessmentConfig(
        source_type=args.source,
        character_path=args.character_path,
        character_id=args.character_id,
        num_tat_images=args.num_tat,
        num_rorschach_cards=args.num_rorschach,
        num_sct_stems=args.num_sct,
        include_rorschach_inquiry=not args.skip_inquiry,
        model_name=args.model,
        use_api=args.use_api,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
    
    # Run assessment
    result = run_single_assessment(config)
    
    # Print summary
    print("\n" + "="*60)
    print("ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Persona: {result.persona_name}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"TAT Responses: {len(result.tat_responses)}")
    print(f"Rorschach Responses: {len(result.rorschach_responses)}")
    print(f"SCT Responses: {len(result.sct_responses)}")
    
    if result.diagnosis:
        if result.diagnosis.get("big_five"):
            bf = result.diagnosis["big_five"]
            print(f"\nBig Five: O={bf.get('O')}, C={bf.get('C')}, E={bf.get('E')}, A={bf.get('A')}, N={bf.get('N')}")
        if result.diagnosis.get("mbti"):
            print(f"MBTI: {result.diagnosis['mbti'].get('type')}")
        if result.diagnosis.get("depression"):
            print(f"Depression Level: {result.diagnosis['depression'].get('level')}")
        if result.diagnosis.get("suicide"):
            print(f"Suicide Risk: {result.diagnosis['suicide'].get('level')}")


if __name__ == "__main__":
    main()
