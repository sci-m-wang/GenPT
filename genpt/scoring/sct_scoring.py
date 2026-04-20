"""
SCT (Sentence Completion Test) Scoring

Implements domain-based scoring for sentence completion responses
across 5 psychological domains.

Domains:
- FA: Family Adjustment
- CA: Career Adjustment
- SA: Self-Attitudes
- IR: Interpersonal Relationships
- ER: Emotion Regulation

Each item is scored 0-6 (0=very positive, 6=severely conflicted).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SCTDomain(Enum):
    """SCT assessment domains."""
    FA = "Family Adjustment"
    CA = "Career Adjustment"
    SA = "Self-Attitudes"
    IR = "Interpersonal Relationships"
    ER = "Emotion Regulation"


# Scoring anchors for SCT items
SCT_SCORING_ANCHORS = {
    0: {
        "label": "Very Positive",
        "description": "Healthy, well-adjusted response showing positive adaptation",
        "examples": [
            "Clear positive affect",
            "Constructive coping",
            "Healthy relationships",
            "Positive self-regard",
        ],
    },
    1: {
        "label": "Positive",
        "description": "Generally positive with minor concerns",
        "examples": [
            "Mostly positive outlook",
            "Minor worries acknowledged",
            "Good adjustment overall",
        ],
    },
    2: {
        "label": "Mildly Positive",
        "description": "Slightly positive, conventional response",
        "examples": [
            "Neutral to slightly positive",
            "Conventional expectations",
            "Adequate functioning",
        ],
    },
    3: {
        "label": "Neutral",
        "description": "Neutral, noncommittal, or stereotyped response",
        "examples": [
            "Neither positive nor negative",
            "Avoidant or evasive",
            "Minimal emotional content",
        ],
    },
    4: {
        "label": "Mildly Negative",
        "description": "Some conflict or negative content",
        "examples": [
            "Mild concerns expressed",
            "Some negative affect",
            "Minor adjustment difficulties",
        ],
    },
    5: {
        "label": "Negative",
        "description": "Significant conflict or distress",
        "examples": [
            "Clear negative affect",
            "Relationship difficulties",
            "Poor self-image",
            "Inadequate coping",
        ],
    },
    6: {
        "label": "Severely Conflicted",
        "description": "Severe maladjustment, pathological content",
        "examples": [
            "Hostile or aggressive content",
            "Hopelessness/helplessness",
            "Severely disturbed relationships",
            "Self-destructive themes",
        ],
    },
}


# Domain-specific scoring guidance
DOMAIN_GUIDANCE = {
    SCTDomain.FA: {
        "focus": "Family relationships and family-of-origin issues",
        "positive_indicators": [
            "Warm family relationships",
            "Positive parental figures",
            "Healthy boundaries",
            "Good family communication",
        ],
        "negative_indicators": [
            "Family conflict",
            "Parental criticism or rejection",
            "Enmeshment or distance",
            "Unresolved childhood issues",
        ],
    },
    SCTDomain.CA: {
        "focus": "Work, career, and achievement orientation",
        "positive_indicators": [
            "Career satisfaction",
            "Achievement motivation",
            "Healthy work-life balance",
            "Positive professional relationships",
        ],
        "negative_indicators": [
            "Work dissatisfaction",
            "Career uncertainty",
            "Conflicts with authority",
            "Fear of failure",
        ],
    },
    SCTDomain.SA: {
        "focus": "Self-concept, identity, and self-esteem",
        "positive_indicators": [
            "Positive self-regard",
            "Self-acceptance",
            "Realistic self-appraisal",
            "Stable identity",
        ],
        "negative_indicators": [
            "Low self-esteem",
            "Self-criticism",
            "Identity confusion",
            "Shame or guilt",
        ],
    },
    SCTDomain.IR: {
        "focus": "Interpersonal relationships and social functioning",
        "positive_indicators": [
            "Trust in others",
            "Satisfying relationships",
            "Social confidence",
            "Healthy intimacy",
        ],
        "negative_indicators": [
            "Social anxiety",
            "Distrust of others",
            "Loneliness",
            "Relationship conflicts",
        ],
    },
    SCTDomain.ER: {
        "focus": "Emotional awareness and regulation",
        "positive_indicators": [
            "Emotional awareness",
            "Adaptive coping",
            "Emotional expression",
            "Stress management",
        ],
        "negative_indicators": [
            "Emotional avoidance",
            "Dysregulation",
            "Overwhelming emotions",
            "Poor stress tolerance",
        ],
    },
}


@dataclass
class ItemScore:
    """Score for a single SCT item."""
    stem_id: str
    stem: str
    completion: str
    domain: SCTDomain
    score: int  # 0-6
    rationale: str = ""
    
    def __post_init__(self):
        # Clamp score to valid range
        self.score = max(0, min(6, self.score))


@dataclass
class DomainScore:
    """Aggregated score for an SCT domain."""
    domain: SCTDomain
    mean_score: float
    item_scores: List[ItemScore] = field(default_factory=list)
    interpretation: str = ""
    
    @property
    def adjustment_level(self) -> str:
        """Get adjustment level label."""
        if self.mean_score < 1.5:
            return "Well-Adjusted"
        elif self.mean_score < 2.5:
            return "Adequately Adjusted"
        elif self.mean_score < 3.5:
            return "Mildly Conflicted"
        elif self.mean_score < 4.5:
            return "Moderately Conflicted"
        else:
            return "Severely Conflicted"


@dataclass
class SCTProfile:
    """Complete SCT scoring profile."""
    domain_scores: Dict[SCTDomain, DomainScore]
    overall_mean: float = 0.0
    
    def __post_init__(self):
        if self.domain_scores:
            self.overall_mean = sum(
                ds.mean_score for ds in self.domain_scores.values()
            ) / len(self.domain_scores)
    
    def to_vector(self) -> List[float]:
        """Convert to ordered domain score vector."""
        return [self.domain_scores[d].mean_score for d in SCTDomain]
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = ["SCT Profile Summary:"]
        for domain in SCTDomain:
            ds = self.domain_scores.get(domain)
            if ds:
                lines.append(f"  {domain.name} ({domain.value}): {ds.mean_score:.2f} - {ds.adjustment_level}")
        lines.append(f"  Overall: {self.overall_mean:.2f}")
        return "\n".join(lines)


class SCTScorer:
    """
    SCT scoring utility.
    
    Provides structured scoring guidelines and aggregation.
    """
    
    DOMAINS = list(SCTDomain)
    SCORE_RANGE = (0, 6)
    
    @staticmethod
    def get_scoring_prompt(stem: str, completion: str, domain: SCTDomain) -> str:
        """Generate a scoring prompt for an SCT item."""
        guidance = DOMAIN_GUIDANCE[domain]
        
        return f"""Score the following sentence completion on a 0-6 scale.

DOMAIN: {domain.value}
Focus: {guidance['focus']}

Positive indicators (score toward 0):
{chr(10).join(f'- {ind}' for ind in guidance['positive_indicators'])}

Negative indicators (score toward 6):
{chr(10).join(f'- {ind}' for ind in guidance['negative_indicators'])}

SCORING SCALE:
0 = Very positive, healthy adjustment
3 = Neutral, noncommittal
6 = Severely conflicted, maladjusted

STEM: {stem}
COMPLETION: {completion}

Provide score (0-6) and brief rationale.
Format: JSON with keys "score" and "rationale"
"""
    
    @staticmethod
    def get_anchor_description(score: int) -> Dict:
        """Get description for a score anchor."""
        return SCT_SCORING_ANCHORS.get(score, SCT_SCORING_ANCHORS[3])
    
    @staticmethod
    def calculate_domain_score(items: List[ItemScore]) -> DomainScore:
        """Calculate domain score from item scores."""
        if not items:
            return DomainScore(
                domain=SCTDomain.SA,
                mean_score=3.0,
                item_scores=[],
                interpretation="No items to score",
            )
        
        domain = items[0].domain
        mean = sum(item.score for item in items) / len(items)
        
        return DomainScore(
            domain=domain,
            mean_score=mean,
            item_scores=items,
            interpretation=SCTScorer.interpret_domain_score(domain, mean),
        )
    
    @staticmethod
    def interpret_domain_score(domain: SCTDomain, score: float) -> str:
        """Provide interpretation for a domain score."""
        guidance = DOMAIN_GUIDANCE[domain]
        
        if score < 2.0:
            return f"Good {domain.value.lower()}. Shows {', '.join(guidance['positive_indicators'][:2])}."
        elif score < 4.0:
            return f"Moderate {domain.value.lower()}. Some mixed indicators."
        else:
            return f"Difficulties in {domain.value.lower()}. Shows {', '.join(guidance['negative_indicators'][:2])}."
    
    @staticmethod
    def create_profile(item_scores: List[ItemScore]) -> SCTProfile:
        """Create complete SCT profile from item scores."""
        # Group by domain
        domain_items = {d: [] for d in SCTDomain}
        for item in item_scores:
            domain_items[item.domain].append(item)
        
        # Calculate domain scores
        domain_scores = {}
        for domain, items in domain_items.items():
            if items:
                domain_scores[domain] = SCTScorer.calculate_domain_score(items)
            else:
                domain_scores[domain] = DomainScore(
                    domain=domain,
                    mean_score=3.0,
                    interpretation="No items for this domain",
                )
        
        return SCTProfile(domain_scores=domain_scores)
