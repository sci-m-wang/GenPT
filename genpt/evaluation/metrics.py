"""
Quantitative metrics for GenPT baselines / adapters / questionnaires.

All four prediction tasks converge on a 3-metric view:
    * macro-F1:  class-balanced F1, robust to label skew
    * accuracy:  exact match
    * MAE:       mean absolute error on the underlying scale

For Big Five we flatten across all 5 traits (O/C/E/A/N each 1-5).
For MBTI we compute dimension-accuracy (matching letters / 4) + type-accuracy
and the MAE of the 4-letter Hamming distance.
For Depression/Suicide (ordinal 0-3) all three metrics are direct.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def macro_f1(preds: Sequence[int], gts: Sequence[int],
             classes: Optional[Sequence[int]] = None) -> float:
    """Class-macro averaged F1 (equal weight per class)."""
    if classes is None:
        classes = sorted(set(list(preds) + list(gts)))
    if not classes:
        return 0.0
    f1s: List[float] = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gts) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def _mae(preds: Sequence[int], gts: Sequence[int]) -> float:
    errors = [abs(int(p) - int(g)) for p, g in zip(preds, gts)]
    return sum(errors) / len(errors) if errors else 0.0


def _accuracy(preds: Sequence[Any], gts: Sequence[Any]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, gts) if p == g) / len(preds)


def compute_big_five_metrics(pairs: List[Tuple[Dict[str, int], Dict[str, int]]]
                              ) -> Dict[str, float]:
    """pairs: list of (pred_dict, gt_dict) where each dict has keys O/C/E/A/N -> 1..5."""
    if not pairs:
        return {"n": 0}
    all_preds: List[int] = []
    all_gts: List[int] = []
    per_dim: Dict[str, float] = {}
    for dim in "OCEAN":
        dp = [int(p.get(dim, 3)) for p, g in pairs]
        dg = [int(g.get(dim, 3)) for p, g in pairs]
        all_preds.extend(dp)
        all_gts.extend(dg)
        per_dim[f"bf_{dim}_f1"] = round(macro_f1(dp, dg, list(range(1, 6))), 3)
    return {
        "bf_n": len(pairs),
        "bf_macro_f1": round(macro_f1(all_preds, all_gts, list(range(1, 6))), 3),
        "bf_accuracy": round(_accuracy(all_preds, all_gts), 3),
        "bf_mae": round(_mae(all_preds, all_gts), 3),
        **per_dim,
        # Exact-5D match: all five traits match
        "bf_exact5": round(
            sum(1 for p, g in pairs
                if all(int(p.get(d, 3)) == int(g.get(d, 3)) for d in "OCEAN"))
            / len(pairs), 3),
    }


def compute_mbti_metrics(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """pairs: list of (pred_type_code, gt_type_code), both 4-letter strings."""
    if not pairs:
        return {"mbti_n": 0}
    total_dims = 0
    matched_dims = 0
    type_matches = 0
    letter_errs: List[int] = []  # "MAE" of Hamming distance / 1
    for p, g in pairs:
        p = (p or "").upper().strip()
        g = (g or "").upper().strip()
        if len(p) == 4 and len(g) == 4:
            total_dims += 4
            m = sum(1 for a, b in zip(p, g) if a == b)
            matched_dims += m
            if m == 4:
                type_matches += 1
            letter_errs.append(4 - m)
        else:
            letter_errs.append(4)
    return {
        "mbti_n": len(pairs),
        "mbti_dim_accuracy": round(matched_dims / total_dims, 3) if total_dims else 0.0,
        "mbti_type_accuracy": round(type_matches / len(pairs), 3),
        "mbti_hamming_mae": round(sum(letter_errs) / len(letter_errs), 3),
    }


def compute_ordinal_metrics(pairs: List[Tuple[int, int]], *,
                             name: str, max_val: int = 3) -> Dict[str, float]:
    """pairs: list of (pred_int, gt_int). name is e.g. 'dep' or 'sui'."""
    if not pairs:
        return {f"{name}_n": 0}
    dp = [int(p) for p, g in pairs]
    dg = [int(g) for p, g in pairs]
    return {
        f"{name}_n": len(pairs),
        f"{name}_macro_f1": round(macro_f1(dp, dg, list(range(max_val + 1))), 3),
        f"{name}_accuracy": round(_accuracy(dp, dg), 3),
        f"{name}_mae": round(_mae(dp, dg), 3),
    }


def compute_all_metrics(predictions: Dict[str, list]) -> Dict[str, Any]:
    """
    predictions: {
        'big_five':   [(pred_dict, gt_dict), ...],
        'mbti':       [(pred_str, gt_str), ...],
        'depression': [(pred_int, gt_int), ...],
        'suicide':    [(pred_int, gt_int), ...],
    }
    """
    metrics: Dict[str, Any] = {}
    if predictions.get("big_five"):
        metrics.update(compute_big_five_metrics(predictions["big_five"]))
    if predictions.get("mbti"):
        metrics.update(compute_mbti_metrics(predictions["mbti"]))
    if predictions.get("depression"):
        metrics.update(compute_ordinal_metrics(predictions["depression"],
                                                name="dep", max_val=3))
    if predictions.get("suicide"):
        metrics.update(compute_ordinal_metrics(predictions["suicide"],
                                                name="sui", max_val=3))
    return metrics


def format_metrics_report(metrics: Dict[str, Any], *,
                          title: str = "Evaluation") -> str:
    """Human-readable report for logging."""
    lines = [f"=== {title} ==="]
    if "bf_n" in metrics:
        lines.append(
            f"Big Five  (n={metrics['bf_n']:3d}): "
            f"macro-F1={metrics['bf_macro_f1']:.3f}  "
            f"acc={metrics['bf_accuracy']:.3f}  "
            f"MAE={metrics['bf_mae']:.3f}  "
            f"exact5={metrics['bf_exact5']:.3f}"
        )
        per_dim = "  ".join(
            f"{d}={metrics[f'bf_{d}_f1']:.2f}" for d in "OCEAN"
            if f"bf_{d}_f1" in metrics
        )
        if per_dim:
            lines.append(f"           per-dim F1: {per_dim}")
    if "mbti_n" in metrics:
        lines.append(
            f"MBTI      (n={metrics['mbti_n']:3d}): "
            f"dim_acc={metrics['mbti_dim_accuracy']:.3f}  "
            f"type_acc={metrics['mbti_type_accuracy']:.3f}  "
            f"hamming_MAE={metrics['mbti_hamming_mae']:.3f}"
        )
    if "dep_n" in metrics:
        lines.append(
            f"Depression(n={metrics['dep_n']:3d}): "
            f"macro-F1={metrics['dep_macro_f1']:.3f}  "
            f"acc={metrics['dep_accuracy']:.3f}  "
            f"MAE={metrics['dep_mae']:.3f}"
        )
    if "sui_n" in metrics:
        lines.append(
            f"Suicide   (n={metrics['sui_n']:3d}): "
            f"macro-F1={metrics['sui_macro_f1']:.3f}  "
            f"acc={metrics['sui_accuracy']:.3f}  "
            f"MAE={metrics['sui_mae']:.3f}"
        )
    return "\n".join(lines)
