"""
GenPT Validity Evaluation

Evaluates the validity of GenPT assessments through:
1. Criterion Validity: Correlation with ground truth labels
2. Construct Validity: Expected personality-score relationships
3. Convergent Validity: Correlation between related measures

Based on the GenPT paper's validity evaluation methodology.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import statistics
from datetime import datetime


# Constants for validity evaluation
BIG_FIVE_TRAITS = ["O", "C", "E", "A", "N"]
MBTI_DIMENSIONS = ["E-I", "S-N", "T-F", "J-P"]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genpt.validity")


@dataclass
class ValidityMetrics:
    """Validity metrics for a specific evaluation."""
    metric_name: str
    correlation: float  # Pearson r or point-biserial
    p_value: Optional[float] = None
    n_samples: int = 0
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant."""
        return self.p_value is not None and self.p_value < alpha


@dataclass
class ValidityReport:
    """Complete validity evaluation report."""
    validity_type: str  # "criterion", "construct", "convergent"
    metrics: List[ValidityMetrics]
    mean_correlation: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def compute_pearson_r(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient and approximate p-value.
    
    Returns:
        Tuple of (correlation, p_value)
    """
    if len(x) != len(y) or len(x) < 3:
        return 0.0, 1.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    den_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    
    if den_x * den_y == 0:
        return 0.0, 1.0
    
    r = num / (den_x * den_y)
    
    # Approximate p-value using Fisher transformation
    # This is a rough approximation
    if abs(r) >= 1:
        p_value = 0.0 if abs(r) == 1 else 1.0
    else:
        import math
        t = r * math.sqrt((n - 2) / (1 - r**2))
        # Approximate p-value (two-tailed)
        # Using approximation: p ≈ 2 * (1 - Φ(|t|)) for large n
        p_value = 2 * (1 - normal_cdf(abs(t)))
    
    return r, p_value


def normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function approximation."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compute_accuracy(predicted: List[Any], actual: List[Any]) -> float:
    """Compute classification accuracy."""
    if len(predicted) != len(actual) or len(predicted) == 0:
        return 0.0
    
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return correct / len(predicted)


def compute_mae(predicted: List[float], actual: List[float]) -> float:
    """Compute Mean Absolute Error."""
    if len(predicted) != len(actual) or len(predicted) == 0:
        return float('inf')
    
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)


def eval_criterion_validity_big_five(
    predicted_scores: List[Dict[str, int]],
    ground_truth: List[Dict[str, int]],
) -> ValidityReport:
    """
    Evaluate criterion validity for Big Five predictions.
    
    Args:
        predicted_scores: List of dicts with O, C, E, A, N scores (1-5)
        ground_truth: List of dicts with true O, C, E, A, N scores
        
    Returns:
        ValidityReport with correlation for each trait
    """
    logger.info(f"Evaluating Big Five criterion validity with {len(predicted_scores)} samples")
    
    metrics = []
    
    for trait in BIG_FIVE_TRAITS:
        pred = [s.get(trait, 3) for s in predicted_scores]
        true = [s.get(trait, 3) for s in ground_truth]
        
        r, p = compute_pearson_r(pred, true)
        mae = compute_mae(pred, true)
        
        metrics.append(ValidityMetrics(
            metric_name=f"Big5_{trait}",
            correlation=r,
            p_value=p,
            n_samples=len(pred),
        ))
    
    mean_r = statistics.mean([m.correlation for m in metrics])
    
    return ValidityReport(
        validity_type="criterion_big_five",
        metrics=metrics,
        mean_correlation=mean_r,
    )


def eval_criterion_validity_mbti(
    predicted_types: List[str],
    ground_truth: List[str],
) -> ValidityReport:
    """
    Evaluate criterion validity for MBTI predictions.
    
    Args:
        predicted_types: List of predicted MBTI types (e.g., "INTJ")
        ground_truth: List of true MBTI types
        
    Returns:
        ValidityReport with accuracy metrics
    """
    logger.info(f"Evaluating MBTI criterion validity with {len(predicted_types)} samples")
    
    metrics = []
    
    # Overall accuracy
    accuracy = compute_accuracy(predicted_types, ground_truth)
    metrics.append(ValidityMetrics(
        metric_name="MBTI_overall_accuracy",
        correlation=accuracy,  # Using accuracy as correlation analog
        n_samples=len(predicted_types),
    ))
    
    # Per-dimension accuracy
    for i, dim in enumerate(MBTI_DIMENSIONS):
        pred_dim = [t[i] if len(t) > i else '?' for t in predicted_types]
        true_dim = [t[i] if len(t) > i else '?' for t in ground_truth]
        
        dim_accuracy = compute_accuracy(pred_dim, true_dim)
        metrics.append(ValidityMetrics(
            metric_name=f"MBTI_{dim}",
            correlation=dim_accuracy,
            n_samples=len(predicted_types),
        ))
    
    mean_acc = statistics.mean([m.correlation for m in metrics])
    
    return ValidityReport(
        validity_type="criterion_mbti",
        metrics=metrics,
        mean_correlation=mean_acc,
    )


def eval_criterion_validity_depression(
    predicted_levels: List[int],
    ground_truth: List[int],
) -> ValidityReport:
    """
    Evaluate criterion validity for depression predictions.
    
    Args:
        predicted_levels: List of predicted depression levels (0-4)
        ground_truth: List of true depression levels
        
    Returns:
        ValidityReport with correlation and accuracy
    """
    logger.info(f"Evaluating depression criterion validity with {len(predicted_levels)} samples")
    
    # Correlation (ordinal)
    r, p = compute_pearson_r(
        [float(x) for x in predicted_levels],
        [float(x) for x in ground_truth]
    )
    
    # Exact accuracy
    accuracy = compute_accuracy(predicted_levels, ground_truth)
    
    # Within-1 accuracy
    within_1 = sum(
        1 for p, t in zip(predicted_levels, ground_truth)
        if abs(p - t) <= 1
    ) / len(predicted_levels) if predicted_levels else 0
    
    metrics = [
        ValidityMetrics(
            metric_name="Depression_correlation",
            correlation=r,
            p_value=p,
            n_samples=len(predicted_levels),
        ),
        ValidityMetrics(
            metric_name="Depression_exact_accuracy",
            correlation=accuracy,
            n_samples=len(predicted_levels),
        ),
        ValidityMetrics(
            metric_name="Depression_within1_accuracy",
            correlation=within_1,
            n_samples=len(predicted_levels),
        ),
    ]
    
    return ValidityReport(
        validity_type="criterion_depression",
        metrics=metrics,
        mean_correlation=r,
    )


def eval_construct_validity(
    assessment_results: List[Dict],
) -> ValidityReport:
    """
    Evaluate construct validity through expected relationships.
    
    Checks theoretically expected correlations:
    - Neuroticism should correlate with depression
    - Extraversion should correlate with social cognition
    - SCORS-G AFF should correlate with interpersonal SCT
    
    Args:
        assessment_results: List of complete assessment results
        
    Returns:
        ValidityReport with construct validity metrics
    """
    logger.info(f"Evaluating construct validity with {len(assessment_results)} samples")
    
    metrics = []
    
    # Extract relevant scores
    neuroticism = []
    depression = []
    extraversion = []
    scors_eir = []  # Emotional Investment in Relationships
    sct_ir = []  # Interpersonal Relations domain
    
    for result in assessment_results:
        diagnosis = result.get("diagnosis", {})
        interpretation = result.get("interpretation", {})
        
        # Big Five
        big_five = diagnosis.get("big_five", {})
        if big_five:
            neuroticism.append(big_five.get("N", 3))
            extraversion.append(big_five.get("E", 3))
        
        # Depression
        dep = diagnosis.get("depression", {})
        if dep:
            depression.append(dep.get("level", 0))
        
        # SCORS-G EIR
        for score in interpretation.get("scors_g_scores", []):
            if score.get("dimension") == "EIR":
                scors_eir.append(score.get("score", 4))
                break
        
        # SCT IR domain
        for score in interpretation.get("sct_scores", []):
            if score.get("domain") == "IR":
                sct_ir.append(score.get("score", 3))
                break
    
    # Test expected relationships
    
    # 1. Neuroticism - Depression (expected positive)
    if len(neuroticism) == len(depression) and len(neuroticism) >= 3:
        r, p = compute_pearson_r(neuroticism, depression)
        metrics.append(ValidityMetrics(
            metric_name="Neuroticism_Depression",
            correlation=r,
            p_value=p,
            n_samples=len(neuroticism),
        ))
    
    # 2. Extraversion - SCORS-G EIR (expected positive)
    if len(extraversion) == len(scors_eir) and len(extraversion) >= 3:
        r, p = compute_pearson_r(extraversion, scors_eir)
        metrics.append(ValidityMetrics(
            metric_name="Extraversion_SCORS_EIR",
            correlation=r,
            p_value=p,
            n_samples=len(extraversion),
        ))
    
    # 3. SCORS-G EIR - SCT IR (convergent validity)
    if len(scors_eir) == len(sct_ir) and len(scors_eir) >= 3:
        r, p = compute_pearson_r(scors_eir, sct_ir)
        metrics.append(ValidityMetrics(
            metric_name="SCORS_EIR_SCT_IR",
            correlation=r,
            p_value=p,
            n_samples=len(scors_eir),
        ))
    
    mean_r = statistics.mean([m.correlation for m in metrics]) if metrics else 0.0
    
    return ValidityReport(
        validity_type="construct",
        metrics=metrics,
        mean_correlation=mean_r,
    )


def eval_convergent_validity(
    genpt_scores: Dict[str, List[float]],
    external_scores: Dict[str, List[float]],
) -> ValidityReport:
    """
    Evaluate convergent validity against external measures.
    
    Args:
        genpt_scores: Dict mapping measure names to GenPT score lists
        external_scores: Dict mapping measure names to external score lists
        
    Returns:
        ValidityReport with convergent validity metrics
    """
    logger.info("Evaluating convergent validity")
    
    metrics = []
    
    for measure in set(genpt_scores.keys()) & set(external_scores.keys()):
        genpt = genpt_scores[measure]
        external = external_scores[measure]
        
        if len(genpt) != len(external) or len(genpt) < 3:
            continue
        
        r, p = compute_pearson_r(genpt, external)
        
        metrics.append(ValidityMetrics(
            metric_name=f"Convergent_{measure}",
            correlation=r,
            p_value=p,
            n_samples=len(genpt),
        ))
    
    mean_r = statistics.mean([m.correlation for m in metrics]) if metrics else 0.0
    
    return ValidityReport(
        validity_type="convergent",
        metrics=metrics,
        mean_correlation=mean_r,
    )


def save_validity_report(report: ValidityReport, output_path: Path) -> None:
    """Save validity report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_dict = {
        "validity_type": report.validity_type,
        "mean_correlation": report.mean_correlation,
        "timestamp": report.timestamp,
        "metrics": [
            {
                "metric_name": m.metric_name,
                "correlation": m.correlation,
                "p_value": m.p_value,
                "n_samples": m.n_samples,
                "is_significant": m.is_significant(),
            }
            for m in report.metrics
        ],
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Validity report saved to: {output_path}")


def print_validity_report(report: ValidityReport) -> None:
    """Print validity report summary."""
    print("\n" + "="*60)
    print(f"VALIDITY REPORT: {report.validity_type.upper()}")
    print("="*60)
    print(f"Mean Correlation: {report.mean_correlation:.3f}")
    print("\nMetric-wise validity:")
    print("-"*50)
    
    for m in report.metrics:
        sig = "*" if m.is_significant() else ""
        print(f"  {m.metric_name}: r={m.correlation:.3f}{sig} (n={m.n_samples})", end="")
        if m.p_value:
            print(f", p={m.p_value:.4f}", end="")
        print()
    
    if any(m.is_significant() for m in report.metrics):
        print("\n* indicates p < 0.05")


def run_full_validity_evaluation(
    assessment_results: List[Dict],
    ground_truth_big_five: Optional[List[Dict]] = None,
    ground_truth_mbti: Optional[List[str]] = None,
    ground_truth_depression: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, ValidityReport]:
    """
    Run full validity evaluation suite.
    
    Args:
        assessment_results: List of complete assessment result dicts
        ground_truth_*: Optional ground truth labels
        output_dir: Optional directory to save reports
        
    Returns:
        Dict mapping validity type to report
    """
    reports = {}
    
    # Criterion validity - Big Five
    if ground_truth_big_five:
        predicted_bf = [
            r.get("diagnosis", {}).get("big_five", {})
            for r in assessment_results
        ]
        report = eval_criterion_validity_big_five(predicted_bf, ground_truth_big_five)
        reports["criterion_big_five"] = report
        print_validity_report(report)
    
    # Criterion validity - MBTI
    if ground_truth_mbti:
        predicted_mbti = [
            r.get("diagnosis", {}).get("mbti", {}).get("type", "????")
            for r in assessment_results
        ]
        report = eval_criterion_validity_mbti(predicted_mbti, ground_truth_mbti)
        reports["criterion_mbti"] = report
        print_validity_report(report)
    
    # Criterion validity - Depression
    if ground_truth_depression:
        predicted_dep = [
            r.get("diagnosis", {}).get("depression", {}).get("level", 0)
            for r in assessment_results
        ]
        report = eval_criterion_validity_depression(predicted_dep, ground_truth_depression)
        reports["criterion_depression"] = report
        print_validity_report(report)
    
    # Construct validity (always run)
    report = eval_construct_validity(assessment_results)
    reports["construct"] = report
    print_validity_report(report)
    
    # Save reports
    if output_dir:
        for name, report in reports.items():
            save_validity_report(report, output_dir / f"validity_{name}.json")
    
    return reports


def run_validity_evaluation(results_dir: str, output_dir: str) -> None:
    """
    Run validity evaluation on saved assessment results.

    Loads result JSONs and ground truth from the results directory,
    then computes criterion and construct validity.

    Args:
        results_dir: Directory containing assessment result JSONs
        output_dir: Directory to save reports
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load result files
    result_files = sorted(results_path.glob("*.json"))
    logger.info(f"Found {len(result_files)} result files in {results_dir}")

    if not result_files:
        logger.warning("No result files found. Run assessments first.")
        return

    assessment_results = []
    gt_big_five = []
    gt_depression = []
    gt_mbti = []

    for rf in result_files:
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        assessment_results.append(data)

        # Try to extract ground truth from metadata if present
        gt = data.get("ground_truth", {})
        if gt.get("big_five"):
            gt_big_five.append(gt["big_five"])
        if gt.get("depression") is not None:
            gt_depression.append(gt["depression"])
        if gt.get("mbti"):
            gt_mbti.append(gt["mbti"])

    run_full_validity_evaluation(
        assessment_results=assessment_results,
        ground_truth_big_five=gt_big_five if gt_big_five else None,
        ground_truth_mbti=gt_mbti if gt_mbti else None,
        ground_truth_depression=gt_depression if gt_depression else None,
        output_dir=output_path,
    )

    logger.info(f"Validity reports saved to {output_path}")
