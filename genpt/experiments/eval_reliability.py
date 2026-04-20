"""
GenPT Reliability Evaluation

Evaluates the reliability of GenPT assessments through:
1. Test-Retest Reliability: Same assessment run multiple times
2. Inter-Rater Reliability: Comparison between different LLM raters
3. Internal Consistency: Cronbach's alpha for scale dimensions

Based on the GenPT paper's reliability evaluation methodology.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import statistics
from datetime import datetime

from ..config import SCORS_G_DIMENSIONS, SRAS_DOMAIN_SCORES, SCT_DOMAINS
from ..llm.qwen import QwenVLClient, QwenTextClient, create_client_from_config
from ..llm.base import BaseLLM, GenerationConfig
from .run_assessment import (
    AssessmentConfig, 
    AssessmentResult,
    run_single_assessment,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genpt.reliability")


@dataclass
class ReliabilityMetrics:
    """Reliability metrics for an assessment dimension."""
    dimension: str
    icc: float  # Intraclass Correlation Coefficient
    cronbach_alpha: Optional[float] = None
    test_retest_r: Optional[float] = None
    inter_rater_kappa: Optional[float] = None
    
    def is_acceptable(self) -> bool:
        """Check if reliability is acceptable (ICC >= 0.7)."""
        return self.icc >= 0.7


@dataclass
class ReliabilityReport:
    """Complete reliability evaluation report."""
    test_type: str  # "test_retest", "inter_rater", "internal_consistency"
    num_trials: int
    dimensions: List[ReliabilityMetrics]
    mean_icc: float
    overall_acceptable: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_scores: Optional[Dict] = None


def compute_icc(scores: List[List[float]]) -> float:
    """
    Compute Intraclass Correlation Coefficient (ICC 2,1).
    
    Args:
        scores: List of score lists, where each inner list represents
                scores from one trial/rater for all items
                
    Returns:
        ICC value between 0 and 1
    """
    if not scores or len(scores) < 2:
        return 0.0
    
    n_trials = len(scores)
    n_items = len(scores[0])
    
    if n_items == 0:
        return 0.0
    
    # Compute means
    trial_means = [statistics.mean(trial) for trial in scores]
    grand_mean = statistics.mean([s for trial in scores for s in trial])
    
    # Compute between-subjects variance (MSB)
    item_means = []
    for i in range(n_items):
        item_scores = [scores[t][i] for t in range(n_trials)]
        item_means.append(statistics.mean(item_scores))
    
    ssb = n_trials * sum((m - grand_mean) ** 2 for m in item_means)
    msb = ssb / (n_items - 1) if n_items > 1 else 0
    
    # Compute within-subjects variance (MSW)
    ssw = sum(
        (scores[t][i] - item_means[i]) ** 2
        for t in range(n_trials)
        for i in range(n_items)
    )
    msw = ssw / (n_items * (n_trials - 1)) if n_trials > 1 else 0
    
    # Compute error variance (MSE)
    sse = sum(
        (scores[t][i] - item_means[i] - trial_means[t] + grand_mean) ** 2
        for t in range(n_trials)
        for i in range(n_items)
    )
    mse = sse / ((n_items - 1) * (n_trials - 1)) if n_items > 1 and n_trials > 1 else 0
    
    # ICC(2,1) formula
    if msb + (n_trials - 1) * mse / n_trials == 0:
        return 0.0
    
    icc = (msb - mse) / (msb + (n_trials - 1) * mse / n_trials + n_trials * (msw - mse) / n_items)
    
    return max(0.0, min(1.0, icc))


def compute_cronbach_alpha(item_scores: List[List[float]]) -> float:
    """
    Compute Cronbach's alpha for internal consistency.
    
    Args:
        item_scores: List where each inner list contains scores for one item
                    across all subjects
                    
    Returns:
        Cronbach's alpha value
    """
    if not item_scores or len(item_scores) < 2:
        return 0.0
    
    n_items = len(item_scores)
    n_subjects = len(item_scores[0])
    
    if n_subjects < 2:
        return 0.0
    
    # Compute item variances
    item_variances = []
    for item in item_scores:
        if len(item) > 1:
            item_variances.append(statistics.variance(item))
        else:
            item_variances.append(0)
    
    # Compute total scores
    total_scores = []
    for s in range(n_subjects):
        total = sum(item_scores[i][s] for i in range(n_items))
        total_scores.append(total)
    
    total_variance = statistics.variance(total_scores) if len(total_scores) > 1 else 0
    
    if total_variance == 0:
        return 0.0
    
    # Cronbach's alpha formula
    alpha = (n_items / (n_items - 1)) * (1 - sum(item_variances) / total_variance)
    
    return max(0.0, min(1.0, alpha))


def compute_pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    den_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    
    if den_x * den_y == 0:
        return 0.0
    
    return num / (den_x * den_y)


def eval_test_retest_reliability(
    config: AssessmentConfig,
    num_trials: int = 3,
    model: Optional[BaseLLM] = None,
) -> ReliabilityReport:
    """
    Evaluate test-retest reliability by running the same assessment multiple times.
    
    Args:
        config: Assessment configuration
        num_trials: Number of repeated assessments
        model: Pre-initialized model (creates new if None)
        
    Returns:
        ReliabilityReport with ICC scores for each dimension
    """
    logger.info(f"Evaluating test-retest reliability with {num_trials} trials")
    
    # Run multiple assessments
    results = []
    for i in range(num_trials):
        logger.info(f"Trial {i+1}/{num_trials}")
        result = run_single_assessment(config, examinee_model=model)
        results.append(result)
    
    # Extract scores by dimension
    dimension_scores = extract_dimension_scores(results)
    
    # Compute reliability metrics
    metrics = []
    for dim, scores in dimension_scores.items():
        icc = compute_icc(scores)
        
        # Compute test-retest correlation (first vs last trial)
        if len(scores) >= 2:
            test_retest_r = compute_pearson_r(scores[0], scores[-1])
        else:
            test_retest_r = None
        
        metrics.append(ReliabilityMetrics(
            dimension=dim,
            icc=icc,
            test_retest_r=test_retest_r,
        ))
    
    mean_icc = statistics.mean([m.icc for m in metrics]) if metrics else 0.0
    
    return ReliabilityReport(
        test_type="test_retest",
        num_trials=num_trials,
        dimensions=metrics,
        mean_icc=mean_icc,
        overall_acceptable=mean_icc >= 0.7,
        raw_scores=dimension_scores,
    )


def eval_inter_rater_reliability(
    config: AssessmentConfig,
    rater_models: List[str],
) -> ReliabilityReport:
    """
    Evaluate inter-rater reliability using different LLM models as raters.
    
    Args:
        config: Assessment configuration
        rater_models: List of model names to use as raters
        
    Returns:
        ReliabilityReport with ICC scores for each dimension
    """
    logger.info(f"Evaluating inter-rater reliability with {len(rater_models)} models")
    
    # Run assessments with different models
    results = []
    for model_name in rater_models:
        logger.info(f"Rater: {model_name}")
        
        # Create model
        from ..config import ModelConfig
        mc = ModelConfig(model_name=model_name, use_vllm=True, api_base=config.api_base, api_key=config.api_key)
        model = create_client_from_config(mc)
        
        # Modify config for this rater
        rater_config = AssessmentConfig(
            **{**config.__dict__, "model_name": model_name}
        )
        
        result = run_single_assessment(rater_config, examinee_model=model)
        results.append(result)
    
    # Extract scores by dimension
    dimension_scores = extract_dimension_scores(results)
    
    # Compute reliability metrics
    metrics = []
    for dim, scores in dimension_scores.items():
        icc = compute_icc(scores)
        metrics.append(ReliabilityMetrics(
            dimension=dim,
            icc=icc,
        ))
    
    mean_icc = statistics.mean([m.icc for m in metrics]) if metrics else 0.0
    
    return ReliabilityReport(
        test_type="inter_rater",
        num_trials=len(rater_models),
        dimensions=metrics,
        mean_icc=mean_icc,
        overall_acceptable=mean_icc >= 0.7,
        raw_scores=dimension_scores,
    )


def eval_internal_consistency(
    results: List[AssessmentResult],
) -> ReliabilityReport:
    """
    Evaluate internal consistency using Cronbach's alpha.
    
    Args:
        results: List of assessment results from different characters
        
    Returns:
        ReliabilityReport with Cronbach's alpha for each scale
    """
    logger.info(f"Evaluating internal consistency with {len(results)} subjects")
    
    # Group scores by dimension across subjects
    scors_g_items = {dim: [] for dim in SCORS_G_DIMENSIONS}
    sct_items = {domain: [] for domain in SCT_DOMAINS}
    
    for result in results:
        if result.interpretation:
            # SCORS-G scores
            for score in result.interpretation.get("scors_g_scores", []):
                dim = score.get("dimension")
                val = score.get("score")
                if dim in scors_g_items and val is not None:
                    scors_g_items[dim].append(val)
            
            # SCT scores
            for score in result.interpretation.get("sct_scores", []):
                domain = score.get("domain")
                val = score.get("score")
                if domain in sct_items and val is not None:
                    sct_items[domain].append(val)
    
    # Compute Cronbach's alpha for each scale
    metrics = []
    
    # SCORS-G alpha
    scors_g_scores_list = list(scors_g_items.values())
    if all(len(s) > 1 for s in scors_g_scores_list):
        alpha = compute_cronbach_alpha(scors_g_scores_list)
        metrics.append(ReliabilityMetrics(
            dimension="SCORS-G",
            icc=alpha,  # Using alpha as ICC equivalent
            cronbach_alpha=alpha,
        ))
    
    # SCT alpha
    sct_scores_list = list(sct_items.values())
    if all(len(s) > 1 for s in sct_scores_list):
        alpha = compute_cronbach_alpha(sct_scores_list)
        metrics.append(ReliabilityMetrics(
            dimension="SCT",
            icc=alpha,
            cronbach_alpha=alpha,
        ))
    
    mean_icc = statistics.mean([m.icc for m in metrics]) if metrics else 0.0
    
    return ReliabilityReport(
        test_type="internal_consistency",
        num_trials=len(results),
        dimensions=metrics,
        mean_icc=mean_icc,
        overall_acceptable=mean_icc >= 0.7,
    )


def extract_dimension_scores(
    results: List[AssessmentResult],
) -> Dict[str, List[List[float]]]:
    """
    Extract scores by dimension from multiple assessment results.
    
    Returns dict mapping dimension names to list of score lists
    (one list per trial/rater).
    """
    scores: Dict[str, List[List[float]]] = {}
    
    for trial_idx, result in enumerate(results):
        if not result.interpretation:
            continue
        
        # TAT/SCORS-G scores (key may be 'tat_scores' or 'scors_g_scores')
        tat_scores_list = (
            result.interpretation.get("tat_scores", [])
            or result.interpretation.get("scors_g_scores", [])
        )
        for score in tat_scores_list:
            dim = f"SCORS-G_{score.get('dimension', '')}"
            val = score.get("score")
            if val is not None:
                scores.setdefault(dim, [[] for _ in range(len(results))])
                if trial_idx < len(scores[dim]):
                    scores[dim][trial_idx].append(float(val))
        
        # Rorschach/SRAS scores
        ror_scores_list = (
            result.interpretation.get("rorschach_scores", [])
            or result.interpretation.get("sras_scores", [])
        )
        for score in ror_scores_list:
            dim = f"SRAS_{score.get('variable', '')}"
            val = score.get("count") or score.get("score")
            if val is not None:
                scores.setdefault(dim, [[] for _ in range(len(results))])
                if trial_idx < len(scores[dim]):
                    scores[dim][trial_idx].append(float(val))
        
        # SCT scores
        for score in result.interpretation.get("sct_scores", []):
            dim = f"SCT_{score.get('domain', '')}"
            val = score.get("score")
            if val is not None:
                scores.setdefault(dim, [[] for _ in range(len(results))])
                if trial_idx < len(scores[dim]):
                    scores[dim][trial_idx].append(float(val))
    
    return scores


def save_reliability_report(report: ReliabilityReport, output_path: Path) -> None:
    """Save reliability report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_dict = {
        "test_type": report.test_type,
        "num_trials": report.num_trials,
        "mean_icc": report.mean_icc,
        "overall_acceptable": report.overall_acceptable,
        "timestamp": report.timestamp,
        "dimensions": [
            {
                "dimension": m.dimension,
                "icc": m.icc,
                "cronbach_alpha": m.cronbach_alpha,
                "test_retest_r": m.test_retest_r,
                "is_acceptable": m.is_acceptable(),
            }
            for m in report.dimensions
        ],
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Reliability report saved to: {output_path}")


def print_reliability_report(report: ReliabilityReport) -> None:
    """Print reliability report summary."""
    print("\n" + "="*60)
    print(f"RELIABILITY REPORT: {report.test_type.upper()}")
    print("="*60)
    print(f"Number of trials: {report.num_trials}")
    print(f"Mean ICC: {report.mean_icc:.3f}")
    print(f"Overall acceptable: {'Yes' if report.overall_acceptable else 'No'}")
    print("\nDimension-wise reliability:")
    print("-"*50)
    
    for m in report.dimensions:
        status = "✓" if m.is_acceptable() else "✗"
        print(f"  {status} {m.dimension}: ICC={m.icc:.3f}", end="")
        if m.cronbach_alpha:
            print(f", α={m.cronbach_alpha:.3f}", end="")
        if m.test_retest_r:
            print(f", r={m.test_retest_r:.3f}", end="")
        print()


def run_reliability_evaluation(results_dir: str, output_dir: str) -> None:
    """
    Run full reliability evaluation on saved assessment results.

    Loads all assessment JSON files from results_dir, computes test-retest
    and internal consistency, and saves reports.

    Args:
        results_dir: Directory containing assessment result JSONs
        output_dir: Directory to save reports
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all result files
    result_files = sorted(results_path.glob("*.json"))
    logger.info(f"Found {len(result_files)} result files in {results_dir}")

    if not result_files:
        logger.warning("No result files found. Run assessments first.")
        return

    results: List[AssessmentResult] = []
    for rf in result_files:
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruct minimal AssessmentResult
        results.append(AssessmentResult(
            config=AssessmentConfig(),
            persona_name=data.get("persona_name", ""),
            source_type=data.get("source_type", ""),
            source_id=data.get("source_id", ""),
            tat_responses=data.get("responses", {}).get("tat", []),
            rorschach_responses=data.get("responses", {}).get("rorschach", []),
            sct_responses=data.get("responses", {}).get("sct", []),
            interpretation=data.get("interpretation"),
            diagnosis=data.get("diagnosis"),
            timestamp=data.get("timestamp", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
        ))

    # Internal consistency
    ic_report = eval_internal_consistency(results)
    save_reliability_report(ic_report, output_path / "internal_consistency.json")
    print_reliability_report(ic_report)

    # Group by persona for test-retest
    persona_groups: Dict[str, List[AssessmentResult]] = {}
    for r in results:
        persona_groups.setdefault(r.persona_name, []).append(r)

    for persona, group in persona_groups.items():
        if len(group) >= 2:
            logger.info(f"Test-retest for {persona} ({len(group)} trials)")
            dim_scores = extract_dimension_scores(group)

            metrics = []
            for dim, s in dim_scores.items():
                icc = compute_icc(s)
                metrics.append(ReliabilityMetrics(dimension=dim, icc=icc))

            mean_icc = statistics.mean([m.icc for m in metrics]) if metrics else 0.0
            report = ReliabilityReport(
                test_type="test_retest",
                num_trials=len(group),
                dimensions=metrics,
                mean_icc=mean_icc,
                overall_acceptable=mean_icc >= 0.7,
            )
            save_reliability_report(report, output_path / f"test_retest_{persona}.json")
            print_reliability_report(report)

    logger.info(f"Reliability reports saved to {output_path}")
