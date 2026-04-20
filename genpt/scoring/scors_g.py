"""
SCORS-G (Social Cognition and Object Relations Scale - Global) Scoring

Implements the 8-dimension SCORS-G framework for TAT narrative analysis.
Each dimension is scored on a 1-7 Likert scale.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SCORSGDimension(Enum):
    """The 8 SCORS-G dimensions."""
    COM = "Complexity of Representations of People"
    AFF = "Affective Quality of Representations"
    EIR = "Emotional Investment in Relationships"
    EIM = "Emotional Investment in Moral Standards"
    SC = "Understanding of Social Causality"
    AGG = "Experience and Management of Aggressive Impulses"
    SE = "Self-Esteem"
    ICS = "Identity and Coherence of Self"


# Dimension descriptions for scoring guidance
SCORS_G_DESCRIPTIONS = {
    SCORSGDimension.COM: {
        "name": "Complexity of Representations of People",
        "low": "People are seen as one-dimensional, undifferentiated, or not mentioned",
        "mid": "Some differentiation of self and others, but limited complexity",
        "high": "People are seen as complex, multidimensional beings with nuanced characteristics",
        "indicators": [
            "Mention of internal states (thoughts, feelings, motivations)",
            "Recognition of conflicting or ambivalent feelings",
            "Differentiation between characters",
            "Understanding of personality traits",
        ],
    },
    SCORSGDimension.AFF: {
        "name": "Affective Quality of Representations",
        "low": "Relationships are malevolent, harmful, or exploitative",
        "mid": "Neutral or mixed expectations about relationships",
        "high": "Relationships are benevolent, nurturing, and supportive",
        "indicators": [
            "Tone of interpersonal interactions",
            "Expectations about how people treat each other",
            "Presence of care, support, or harm",
            "Emotional safety in relationships",
        ],
    },
    SCORSGDimension.EIR: {
        "name": "Emotional Investment in Relationships",
        "low": "No emotional investment; relationships are instrumental or absent",
        "mid": "Some investment but shallow or conditional",
        "high": "Deep, genuine emotional investment in relationships",
        "indicators": [
            "Importance of relationships to characters",
            "Emotional depth in interactions",
            "Concern for others' wellbeing",
            "Intimacy and connection",
        ],
    },
    SCORSGDimension.EIM: {
        "name": "Emotional Investment in Moral Standards",
        "low": "No moral framework; amoral or antisocial behavior",
        "mid": "External morality; rules followed due to consequences",
        "high": "Internalized values; genuine moral concern and empathy",
        "indicators": [
            "References to right/wrong",
            "Guilt or remorse",
            "Ethical decision-making",
            "Concern for fairness and justice",
        ],
    },
    SCORSGDimension.SC: {
        "name": "Understanding of Social Causality",
        "low": "No understanding of cause-effect in social situations",
        "mid": "Basic understanding but oversimplified",
        "high": "Sophisticated understanding of complex social dynamics",
        "indicators": [
            "Explanation of why events occur",
            "Understanding motivations",
            "Recognition of multiple causes",
            "Psychological insight",
        ],
    },
    SCORSGDimension.AGG: {
        "name": "Experience and Management of Aggressive Impulses",
        "low": "Uncontrolled aggression; violence without consequence",
        "mid": "Aggression present but with some restraint",
        "high": "Aggression acknowledged and managed constructively",
        "indicators": [
            "How anger/aggression is expressed",
            "Consequences of aggressive behavior",
            "Resolution of conflicts",
            "Sublimation of aggressive impulses",
        ],
    },
    SCORSGDimension.SE: {
        "name": "Self-Esteem",
        "low": "Characters are worthless, incompetent, or deeply flawed",
        "mid": "Mixed or fragile self-worth",
        "high": "Characters have stable, positive self-regard",
        "indicators": [
            "How protagonists view themselves",
            "Confidence in abilities",
            "Response to failure",
            "Self-acceptance",
        ],
    },
    SCORSGDimension.ICS: {
        "name": "Identity and Coherence of Self",
        "low": "Fragmented, confused, or absent sense of identity",
        "mid": "Some identity but prone to disruption",
        "high": "Stable, coherent sense of self that persists",
        "indicators": [
            "Consistency of character across story",
            "Sense of personal continuity",
            "Goals and values",
            "Response to challenges to identity",
        ],
    },
}


# Anchor scoring guidelines (1-7 scale)
SCORING_ANCHORS = {
    1: "Severe impairment; pathological functioning",
    2: "Significant impairment; poor functioning",
    3: "Moderate impairment; below average functioning",
    4: "Average; normative functioning",
    5: "Above average; good functioning",
    6: "High functioning; well-developed capacity",
    7: "Exceptionally high; optimal functioning",
}


@dataclass
class DimensionScore:
    """Score for a single SCORS-G dimension."""
    dimension: SCORSGDimension
    score: int  # 1-7
    evidence: List[str] = field(default_factory=list)
    rationale: str = ""
    
    def __post_init__(self):
        # Clamp score to valid range
        self.score = max(1, min(7, self.score))


@dataclass
class SCORSGProfile:
    """Complete SCORS-G profile for a narrative."""
    narrative_id: str
    dimension_scores: Dict[SCORSGDimension, DimensionScore]
    
    def get_score(self, dim: SCORSGDimension) -> int:
        """Get score for a specific dimension."""
        return self.dimension_scores.get(dim, DimensionScore(dim, 4)).score
    
    def to_vector(self) -> List[int]:
        """Convert to ordered score vector."""
        return [self.get_score(dim) for dim in SCORSGDimension]
    
    def mean_score(self) -> float:
        """Calculate mean across all dimensions."""
        scores = [ds.score for ds in self.dimension_scores.values()]
        return sum(scores) / len(scores) if scores else 4.0
    
    def summary(self) -> str:
        """Generate a text summary."""
        lines = [f"SCORS-G Profile for {self.narrative_id}:"]
        for dim in SCORSGDimension:
            ds = self.dimension_scores.get(dim)
            if ds:
                lines.append(f"  {dim.name}: {ds.score}/7 - {ds.rationale[:50]}...")
            else:
                lines.append(f"  {dim.name}: N/A")
        return "\n".join(lines)


class SCORSGScorer:
    """
    SCORS-G scoring utility.
    
    Provides structured scoring guidelines and validation.
    """
    
    DIMENSIONS = list(SCORSGDimension)
    SCORE_RANGE = (1, 7)
    
    @staticmethod
    def get_dimension_info(dim: SCORSGDimension) -> Dict:
        """Get detailed information about a dimension."""
        return SCORS_G_DESCRIPTIONS.get(dim, {})
    
    @staticmethod
    def get_scoring_prompt(dim: SCORSGDimension, narrative: str) -> str:
        """Generate a scoring prompt for a specific dimension."""
        info = SCORS_G_DESCRIPTIONS[dim]
        
        return f"""Score the following narrative on the SCORS-G dimension: {dim.value}

DIMENSION: {info['name']}
- Low (1-2): {info['low']}
- Mid (3-5): {info['mid']}
- High (6-7): {info['high']}

Key indicators to look for:
{chr(10).join(f'- {ind}' for ind in info['indicators'])}

NARRATIVE:
{narrative}

Provide:
1. Score (1-7)
2. Specific textual evidence from the narrative
3. Rationale for the score

Format: JSON with keys "score", "evidence", "rationale"
"""
    
    @staticmethod
    def validate_score(score: int) -> bool:
        """Validate that a score is in the valid range."""
        return 1 <= score <= 7
    
    @staticmethod
    def aggregate_profiles(profiles: List[SCORSGProfile]) -> Dict[SCORSGDimension, float]:
        """Aggregate multiple profiles into mean scores per dimension."""
        if not profiles:
            return {dim: 4.0 for dim in SCORSGDimension}
        
        totals = {dim: [] for dim in SCORSGDimension}
        for profile in profiles:
            for dim, ds in profile.dimension_scores.items():
                totals[dim].append(ds.score)
        
        return {
            dim: sum(scores) / len(scores) if scores else 4.0
            for dim, scores in totals.items()
        }
    
    @staticmethod
    def interpret_score(dim: SCORSGDimension, score: float) -> str:
        """Provide clinical interpretation of a score."""
        info = SCORS_G_DESCRIPTIONS[dim]
        
        if score < 2.5:
            level = "low"
            desc = info["low"]
        elif score < 5.5:
            level = "mid"
            desc = info["mid"]
        else:
            level = "high"
            desc = info["high"]
        
        return f"{dim.value}: {level.upper()} ({score:.1f}/7) - {desc}"
