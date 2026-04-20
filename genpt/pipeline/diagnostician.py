"""
Diagnostician Module (Stage 3: Diagnosis)

The Diagnostician aggregates structured indicators from the Interpreter
to produce final psychological state predictions.

Equation: ŷ = D(S, E)

Supports:
- Personality prediction: Big Five (1-5 levels), MBTI (continuous scores)
- Mental health prediction: Depression risk (0-4), Suicide risk (0-3)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json
import re

from ..llm.base import BaseLLM, Message, GenerationConfig
from ..config import BIG_FIVE_TASK, MBTI_TASK, DEPRESSION_TASK, SUICIDE_TASK
from .interpreter import InterpretationResult, SCORSGScore, SRASScore, SCTScore


# ============================================================
# Prediction Data Classes
# ============================================================

@dataclass
class BigFivePrediction:
    """Big Five personality prediction."""
    openness: int  # 1-5
    conscientiousness: int  # 1-5
    extraversion: int  # 1-5
    agreeableness: int  # 1-5
    neuroticism: int  # 1-5
    explanations: Dict[str, str] = field(default_factory=dict)
    
    def to_vector(self) -> List[int]:
        return [self.openness, self.conscientiousness, self.extraversion, 
                self.agreeableness, self.neuroticism]
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "O": self.openness,
            "C": self.conscientiousness,
            "E": self.extraversion,
            "A": self.agreeableness,
            "N": self.neuroticism,
        }


@dataclass
class MBTIPrediction:
    """MBTI personality prediction with continuous scores."""
    ei_score: float  # 0=E, 1=I
    sn_score: float  # 0=S, 1=N
    tf_score: float  # 0=T, 1=F
    jp_score: float  # 0=J, 1=P
    explanations: Dict[str, str] = field(default_factory=dict)
    
    @property
    def type_code(self) -> str:
        """Get the discrete MBTI type code."""
        e_i = "I" if self.ei_score > 0.5 else "E"
        s_n = "N" if self.sn_score > 0.5 else "S"
        t_f = "F" if self.tf_score > 0.5 else "T"
        j_p = "P" if self.jp_score > 0.5 else "J"
        return f"{e_i}{s_n}{t_f}{j_p}"
    
    def to_vector(self) -> List[float]:
        return [self.ei_score, self.sn_score, self.tf_score, self.jp_score]


@dataclass
class DepressionPrediction:
    """Depression risk level prediction (0-3, aligned with 4-level schema)."""
    level: int  # 0=minimal, 1=mild, 2=moderate, 3=severe
    confidence: float = 0.0
    explanation: str = ""

    LEVEL_LABELS = ["Minimal", "Mild", "Moderate", "Severe"]

    @property
    def label(self) -> str:
        return self.LEVEL_LABELS[min(self.level, 3)]


@dataclass
class SuicidePrediction:
    """Suicide ideation risk prediction."""
    level: int  # 0=none, 1=low, 2=moderate, 3=high
    confidence: float = 0.0
    explanation: str = ""
    
    LEVEL_LABELS = ["None", "Low", "Moderate", "High"]
    
    @property
    def label(self) -> str:
        return self.LEVEL_LABELS[min(self.level, 3)]


@dataclass
class DiagnosisResult:
    """Complete diagnosis results."""
    big_five: Optional[BigFivePrediction] = None
    mbti: Optional[MBTIPrediction] = None
    depression: Optional[DepressionPrediction] = None
    suicide: Optional[SuicidePrediction] = None


# ============================================================
# Diagnostician Class
# ============================================================

class Diagnostician:
    """
    Diagnostician class for Stage 3 diagnosis.
    
    Aggregates structured indicators to produce final psychological state predictions.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the Diagnostician.
        
        Args:
            llm: The LLM model to use
            generation_config: Optional generation settings
        """
        self.llm = llm
        self.config = generation_config or GenerationConfig(
            max_tokens=2048,
            temperature=0.3,
        )
    
    # ========================================
    # Personality Prediction
    # ========================================
    
    def predict_big_five(
        self,
        interpretation: InterpretationResult,
    ) -> BigFivePrediction:
        """
        Predict Big Five personality levels.
        
        Args:
            interpretation: Complete interpretation results
            
        Returns:
            BigFivePrediction with 5 dimension levels (1-5)
        """
        # Build structured input for the LLM
        scores_summary = self._build_scores_summary(interpretation)
        
        prompt = f"""Based on the following psychological assessment scores, predict the Big Five personality levels.

{scores_summary}

For each Big Five dimension, predict the level from 1-5:
- 1: Very Low
- 2: Low
- 3: Moderate
- 4: High
- 5: Very High

Output JSON:
{{
    "O": {{"level": <1-5>, "explanation": "<reason>"}},
    "C": {{"level": <1-5>, "explanation": "<reason>"}},
    "E": {{"level": <1-5>, "explanation": "<reason>"}},
    "A": {{"level": <1-5>, "explanation": "<reason>"}},
    "N": {{"level": <1-5>, "explanation": "<reason>"}}
}}"""
        
        result = self.llm.chat(
            user_message=prompt,
            system_prompt=DIAGNOSTICIAN_SYSTEM_PROMPT,
            config=self.config,
        )
        
        return self._parse_big_five_output(result)
    
    def predict_mbti(
        self,
        interpretation: InterpretationResult,
    ) -> MBTIPrediction:
        """
        Predict MBTI with continuous scores.
        
        Args:
            interpretation: Complete interpretation results
            
        Returns:
            MBTIPrediction with 4 dimension scores (0-1)
        """
        scores_summary = self._build_scores_summary(interpretation)
        
        prompt = f"""Based on the following psychological assessment scores, predict MBTI dimensions.

{scores_summary}

For each MBTI dimension, predict a continuous score from 0.0 to 1.0:
- E-I: 0=Extraversion, 1=Introversion
- S-N: 0=Sensing, 1=Intuition
- T-F: 0=Thinking, 1=Feeling
- J-P: 0=Judging, 1=Perceiving

Output JSON:
{{
    "EI": {{"score": <0.0-1.0>, "explanation": "<reason>"}},
    "SN": {{"score": <0.0-1.0>, "explanation": "<reason>"}},
    "TF": {{"score": <0.0-1.0>, "explanation": "<reason>"}},
    "JP": {{"score": <0.0-1.0>, "explanation": "<reason>"}}
}}"""
        
        result = self.llm.chat(
            user_message=prompt,
            system_prompt=DIAGNOSTICIAN_SYSTEM_PROMPT,
            config=self.config,
        )
        
        return self._parse_mbti_output(result)
    
    # ========================================
    # Mental Health Prediction
    # ========================================
    
    def predict_depression(
        self,
        interpretation: InterpretationResult,
    ) -> DepressionPrediction:
        """
        Predict depression risk level.
        
        Args:
            interpretation: Complete interpretation results
            
        Returns:
            DepressionPrediction with level (0-4)
        """
        scores_summary = self._build_scores_summary(interpretation)
        
        prompt = f"""Based on the following psychological assessment scores, predict the depression risk level.

{scores_summary}

Depression Risk Levels:
- 0: Minimal - No significant depressive symptoms
- 1: Mild - Some symptoms, minor impairment
- 2: Moderate - Clear symptoms, noticeable impairment
- 3: Moderately Severe - Many symptoms, significant impairment
- 4: Severe - Most symptoms present, severe impairment

Consider indicators such as:
- Negative affect and hopelessness
- Self-esteem and self-attitudes
- Social withdrawal
- Emotional dysregulation

Output JSON:
{{
    "level": <0-4>,
    "confidence": <0.0-1.0>,
    "explanation": "<detailed reasoning>"
}}"""
        
        result = self.llm.chat(
            user_message=prompt,
            system_prompt=DIAGNOSTICIAN_SYSTEM_PROMPT,
            config=self.config,
        )
        
        return self._parse_depression_output(result)
    
    def predict_suicide(
        self,
        interpretation: InterpretationResult,
    ) -> SuicidePrediction:
        """
        Predict suicide ideation risk level.
        
        Args:
            interpretation: Complete interpretation results
            
        Returns:
            SuicidePrediction with level (0-3)
        """
        scores_summary = self._build_scores_summary(interpretation)
        
        prompt = f"""Based on the following psychological assessment scores, predict the suicide ideation risk level.

{scores_summary}

Suicide Risk Levels:
- 0: None - No indicators of suicidal ideation
- 1: Low - Passive ideation, no plan or intent
- 2: Moderate - Active ideation with some planning
- 3: High - Clear ideation with plan and/or intent

Consider indicators such as:
- Hopelessness and helplessness themes
- Morbid or death-related content
- Self-destructive patterns
- Social isolation themes

Output JSON:
{{
    "level": <0-3>,
    "confidence": <0.0-1.0>,
    "explanation": "<detailed reasoning>"
}}"""
        
        result = self.llm.chat(
            user_message=prompt,
            system_prompt=DIAGNOSTICIAN_SYSTEM_PROMPT,
            config=self.config,
        )
        
        return self._parse_suicide_output(result)
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _build_scores_summary(self, interpretation: InterpretationResult) -> str:
        """Build a text summary of all interpretation scores."""
        lines = []
        
        # TAT SCORS-G scores
        lines.append("=== TAT Analysis (SCORS-G) ===")
        aggregated = interpretation.get_aggregated_tat_scores()
        for dim, score in aggregated.items():
            lines.append(f"  {dim}: {score:.2f}")
        
        # Rorschach SRAS scores
        lines.append("\n=== Rorschach Analysis (SRAS) ===")
        sras = interpretation.rorschach_scores
        lines.append(f"  CPS (Cognitive Processing): {sras.cps:.2f}")
        lines.append(f"  ARS (Affective Regulation): {sras.ars:.2f}")
        lines.append(f"  IRS (Interpersonal Relations): {sras.irs:.2f}")
        lines.append(f"  SCS (Stress Coping): {sras.scs:.2f}")
        
        # SCT domain scores
        lines.append("\n=== SCT Analysis ===")
        for domain, score in interpretation.sct_scores.domain_scores.items():
            lines.append(f"  {domain}: {score:.2f}")
        
        return "\n".join(lines)
    
    def _parse_big_five_output(self, output: str) -> BigFivePrediction:
        """Parse Big Five prediction output."""
        defaults = BigFivePrediction(3, 3, 3, 3, 3)
        explanations = {}
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                data = json.loads(json_match.group())
                defaults = BigFivePrediction(
                    openness=int(data.get("O", {}).get("level", 3)),
                    conscientiousness=int(data.get("C", {}).get("level", 3)),
                    extraversion=int(data.get("E", {}).get("level", 3)),
                    agreeableness=int(data.get("A", {}).get("level", 3)),
                    neuroticism=int(data.get("N", {}).get("level", 3)),
                    explanations={
                        k: v.get("explanation", "") 
                        for k, v in data.items() if isinstance(v, dict)
                    },
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        return defaults
    
    def _parse_mbti_output(self, output: str) -> MBTIPrediction:
        """Parse MBTI prediction output."""
        defaults = MBTIPrediction(0.5, 0.5, 0.5, 0.5)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                data = json.loads(json_match.group())
                defaults = MBTIPrediction(
                    ei_score=float(data.get("EI", {}).get("score", 0.5)),
                    sn_score=float(data.get("SN", {}).get("score", 0.5)),
                    tf_score=float(data.get("TF", {}).get("score", 0.5)),
                    jp_score=float(data.get("JP", {}).get("score", 0.5)),
                    explanations={
                        k: v.get("explanation", "") 
                        for k, v in data.items() if isinstance(v, dict)
                    },
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        
        return defaults
    
    def _parse_depression_output(self, output: str) -> DepressionPrediction:
        """Parse depression prediction output."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                data = json.loads(json_match.group())
                return DepressionPrediction(
                    level=int(data.get("level", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    explanation=data.get("explanation", ""),
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        
        return DepressionPrediction(0, 0.5, "Unable to parse prediction")
    
    def _parse_suicide_output(self, output: str) -> SuicidePrediction:
        """Parse suicide prediction output."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                data = json.loads(json_match.group())
                return SuicidePrediction(
                    level=int(data.get("level", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    explanation=data.get("explanation", ""),
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        
        return SuicidePrediction(0, 0.5, "Unable to parse prediction")
    
    # ========================================
    # Full Diagnosis
    # ========================================
    
    def diagnose_all(
        self,
        interpretation: InterpretationResult,
        predict_personality: bool = True,
        predict_mental_health: bool = True,
    ) -> DiagnosisResult:
        """
        Run complete diagnosis on interpretation results.
        
        Args:
            interpretation: Complete interpretation results
            predict_personality: Whether to predict Big Five and MBTI
            predict_mental_health: Whether to predict depression and suicide risk
            
        Returns:
            DiagnosisResult with all predictions
        """
        result = DiagnosisResult()
        
        if predict_personality:
            result.big_five = self.predict_big_five(interpretation)
            result.mbti = self.predict_mbti(interpretation)
        
        if predict_mental_health:
            result.depression = self.predict_depression(interpretation)
            result.suicide = self.predict_suicide(interpretation)
        
        return result


# ============================================================
# System Prompts
# ============================================================

DIAGNOSTICIAN_SYSTEM_PROMPT = """You are an expert clinical psychologist serving as the Diagnostician in a psychological assessment system.

Your role is to integrate multiple sources of psychological data to make predictions about personality traits and mental health status.

Guidelines:
1. Consider all available indicators holistically
2. Weight projective test indicators according to their reliability
3. Look for convergent evidence across different tests
4. Be conservative in predictions when evidence is ambiguous
5. Provide clear reasoning for your predictions

Remember: This is for research purposes only. These predictions are not clinical diagnoses."""
