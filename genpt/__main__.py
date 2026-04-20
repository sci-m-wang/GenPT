"""
GenPT CLI entry point.

Usage:
    python -m genpt assess ...
    python -m genpt eval ...
    python -m genpt stimuli ...
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="genpt",
        description="GenPT: Generative Projective Testing for LLM Psychological Assessment",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- assess ----
    assess_parser = subparsers.add_parser("assess", help="Run psychological assessment")
    assess_parser.add_argument(
        "--source",
        choices=["characterrag", "annaagent"],
        default="characterrag",
        help="Character data source",
    )
    assess_parser.add_argument(
        "--character-path", required=True, help="Path to character file/directory"
    )
    assess_parser.add_argument("--character-id", help="Character ID (for AnnaAgent)")
    assess_parser.add_argument(
        "--model", default=None, help="Model name (default from config)"
    )
    assess_parser.add_argument("--use-api", action="store_true", help="Use API mode")
    assess_parser.add_argument("--api-base", help="API base URL")
    assess_parser.add_argument("--api-key", help="API key")
    assess_parser.add_argument(
        "--output-dir", default="./results", help="Output directory"
    )
    assess_parser.add_argument(
        "--num-tat", type=int, default=8, help="TAT images (default 8)"
    )
    assess_parser.add_argument(
        "--num-sct", type=int, default=20, help="SCT stems (default 20)"
    )
    assess_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Run evaluation experiments")
    eval_parser.add_argument(
        "--experiment",
        required=True,
        choices=["reliability", "validity"],
        help="Experiment to run",
    )
    eval_parser.add_argument("--results-dir", required=True, help="Results directory")
    eval_parser.add_argument(
        "--output-dir", default="./eval_output", help="Output directory"
    )

    # ---- stimuli ----
    stimuli_parser = subparsers.add_parser("stimuli", help="Inspect loaded stimuli")
    stimuli_parser.add_argument(
        "--type",
        choices=["tat", "rorschach", "sct", "all"],
        default="all",
        help="Stimuli type to inspect",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "assess":
        _run_assess(args)
    elif args.command == "eval":
        _run_eval(args)
    elif args.command == "stimuli":
        _run_stimuli(args)


def _run_assess(args):
    from .config import DEFAULT_MODEL_CONFIG
    from .experiments.run_assessment import AssessmentConfig, run_single_assessment

    config = AssessmentConfig(
        source_type=args.source,
        character_path=args.character_path,
        character_id=args.character_id,
        num_tat_images=args.num_tat,
        num_sct_stems=args.num_sct,
        model_name=args.model or DEFAULT_MODEL_CONFIG.model_name,
        use_api=args.use_api,
        api_base=args.api_base,
        api_key=args.api_key,
        output_dir=args.output_dir,
    )
    result = run_single_assessment(config)
    print(
        f"\nAssessment complete for {result.persona_name} ({result.duration_seconds:.1f}s)"
    )
    if result.diagnosis:
        if result.diagnosis.get("big_five"):
            bf = result.diagnosis["big_five"]
            print(
                f"Big Five: O={bf.get('O')} C={bf.get('C')} E={bf.get('E')} A={bf.get('A')} N={bf.get('N')}"
            )


def _run_eval(args):
    print(f"[eval] Experiment: {args.experiment}")
    if args.experiment == "reliability":
        from .experiments.eval_reliability import run_reliability_evaluation

        run_reliability_evaluation(args.results_dir, args.output_dir)
    elif args.experiment == "validity":
        from .experiments.eval_validity import run_validity_evaluation

        run_validity_evaluation(args.results_dir, args.output_dir)


def _run_stimuli(args):
    from .stimuli.tat import TATStimuli
    from .stimuli.rorschach import RorschachStimuli
    from .stimuli.sct import SCTStimuli

    if args.type in ("tat", "all"):
        tat = TATStimuli()
        print(tat.summary())
        print()
    if args.type in ("rorschach", "all"):
        ror = RorschachStimuli()
        print(ror.summary())
        print()
    if args.type in ("sct", "all"):
        sct = SCTStimuli()
        print(sct.summary())
        print()


if __name__ == "__main__":
    main()
