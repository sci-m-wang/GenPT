"""
Interpreter Module (Stage 2: Interpretation)

The Interpreter transforms unstructured Examinee responses into structured 
psychological indicators using standardized frameworks:
- SCORS-G for TAT narratives
- SRAS for Rorschach responses
- Domain scoring for SCT completions

Equation: s_i, E_i = I(r_i)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json
import re

from ..llm.base import BaseLLM, Message, GenerationConfig
from ..config import SCORS_G_DIMENSIONS, SCORS_G_SCORE_RANGE, SRAS_DOMAIN_SCORES, SCT_DOMAINS
from .examinee import TATResponse, RorschachResponse, SCTResponse


# ============================================================
# Analysis Result Data Classes
# ============================================================

@dataclass
class SCORSGScore:
    """SCORS-G scores for a single TAT narrative."""
    response_id: str
    scores: Dict[str, int]  # {dimension: score (1-7)}
    explanations: Dict[str, str]  # {dimension: explanation}
    
    def to_vector(self) -> List[int]:
        """Convert to ordered score vector."""
        return [self.scores.get(dim, 4) for dim in SCORS_G_DIMENSIONS]
    
    def mean_score(self) -> float:
        """Calculate mean across all dimensions."""
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0


@dataclass
class SRASScore:
    """SRAS scores for Rorschach responses."""
    # Raw encoding counts
    encoding: Dict[str, int] = field(default_factory=dict)
    
    # Domain scores
    cps: float = 0.0  # Cognitive Processing Score
    ars: float = 0.0  # Affective Regulation Score
    irs: float = 0.0  # Interpersonal Relations Score
    scs: float = 0.0  # Stress Coping Score
    
    explanations: Dict[str, str] = field(default_factory=dict)
    
    def to_vector(self) -> List[float]:
        """Convert to score vector."""
        return [self.cps, self.ars, self.irs, self.scs]


@dataclass
class SCTScore:
    """SCT scores for sentence completions."""
    domain_scores: Dict[str, float]  # {domain: mean score (0-6)}
    item_scores: Dict[str, int]  # {stem_id: score (0-6)}
    explanations: Dict[str, str]  # {stem_id or domain: explanation}
    
    def to_vector(self) -> List[float]:
        """Convert to ordered domain score vector."""
        domains = ["FA", "CA", "SA", "IR", "ER"]
        return [self.domain_scores.get(d, 3.0) for d in domains]


@dataclass
class InterpretationResult:
    """Complete interpretation results for all projective tests."""
    tat_scores: List[SCORSGScore]
    rorschach_scores: SRASScore
    sct_scores: SCTScore
    
    def get_aggregated_tat_scores(self) -> Dict[str, float]:
        """Get mean SCORS-G scores across all TAT responses."""
        if not self.tat_scores:
            return {dim: 4.0 for dim in SCORS_G_DIMENSIONS}
        
        aggregated = {dim: 0.0 for dim in SCORS_G_DIMENSIONS}
        for score in self.tat_scores:
            for dim, val in score.scores.items():
                aggregated[dim] += val
        
        n = len(self.tat_scores)
        return {dim: val / n for dim, val in aggregated.items()}


# ============================================================
# Interpreter Class
# ============================================================

class Interpreter:
    """
    Interpreter class for Stage 2 analysis.
    
    Transforms unstructured responses into structured psychological indicators.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the Interpreter.
        
        Args:
            llm: The LLM model to use for analysis
            generation_config: Optional generation settings
        """
        self.llm = llm
        self.config = generation_config or GenerationConfig(
            max_tokens=2048,
            temperature=0.3,  # Lower temperature for more consistent scoring
        )
    
    # ========================================
    # TAT Analysis (SCORS-G)
    # ========================================
    
    def analyze_tat(
        self,
        response: TATResponse,
    ) -> SCORSGScore:
        """
        Analyze a TAT narrative using SCORS-G framework.
        
        Args:
            response: The TAT narrative response
            
        Returns:
            SCORSGScore with 8 dimension scores and explanations
        """
        prompt = self._build_scors_g_prompt(response.narrative)
        
        result = self.llm.chat(
            user_message=prompt,
            system_prompt=SCORS_G_SYSTEM_PROMPT,
            config=self.config,
        )
        
        # Parse the structured output
        scores, explanations = self._parse_scors_g_output(result)
        
        return SCORSGScore(
            response_id=response.image_id,
            scores=scores,
            explanations=explanations,
        )
    
    def analyze_tat_batch(
        self,
        responses: List[TATResponse],
    ) -> List[SCORSGScore]:
        """Analyze multiple TAT narratives."""
        return [self.analyze_tat(r) for r in responses]
    
    def _build_scors_g_prompt(self, narrative: str) -> str:
        """Build SCORS-G analysis prompt."""
        return f"""Analyze the following TAT narrative using the SCORS-G framework.

NARRATIVE:
{narrative}

For each of the 8 SCORS-G dimensions, provide:
1. A score from 1-7 (1=lowest, 7=highest)
2. A brief explanation for your score

Output in JSON format:
{{
    "COM": {{"score": <1-7>, "explanation": "<reason>"}},
    "AFF": {{"score": <1-7>, "explanation": "<reason>"}},
    "EIR": {{"score": <1-7>, "explanation": "<reason>"}},
    "EIM": {{"score": <1-7>, "explanation": "<reason>"}},
    "SC": {{"score": <1-7>, "explanation": "<reason>"}},
    "AGG": {{"score": <1-7>, "explanation": "<reason>"}},
    "SE": {{"score": <1-7>, "explanation": "<reason>"}},
    "ICS": {{"score": <1-7>, "explanation": "<reason>"}}
}}"""
    
    def _parse_scors_g_output(self, output: str) -> Tuple[Dict[str, int], Dict[str, str]]:
        """Parse SCORS-G JSON output."""
        scores = {}
        explanations = {}
        
        try:
            # Extract JSON from output
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                data = json.loads(json_match.group())
                for dim in SCORS_G_DIMENSIONS:
                    if dim in data:
                        scores[dim] = int(data[dim].get("score", 4))
                        explanations[dim] = data[dim].get("explanation", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            # Default to middle scores if parsing fails
            for dim in SCORS_G_DIMENSIONS:
                scores[dim] = 4
                explanations[dim] = "Unable to parse response"
        
        return scores, explanations
    
    # ========================================
    # Rorschach Analysis (SRAS)
    # ========================================
    
    def analyze_rorschach(
        self,
        responses: List[RorschachResponse],
    ) -> SRASScore:
        """
        Analyze Rorschach responses using SRAS framework.
        
        Args:
            responses: List of Rorschach card responses
            
        Returns:
            SRASScore with domain scores and explanations
        """
        # First, encode all responses
        encoding = self._encode_rorschach_responses(responses)
        
        # Calculate domain scores using SRAS formulas
        cps = self._calculate_cps(encoding)
        ars = self._calculate_ars(encoding)
        irs = self._calculate_irs(encoding)
        scs = self._calculate_scs(encoding)
        
        # Generate explanations
        explanations = self._generate_sras_explanations(encoding, cps, ars, irs, scs)
        
        return SRASScore(
            encoding=encoding,
            cps=cps,
            ars=ars,
            irs=irs,
            scs=scs,
            explanations=explanations,
        )
    
    def _encode_rorschach_responses(
        self,
        responses: List[RorschachResponse],
    ) -> Dict[str, int]:
        """Encode Rorschach responses into variable counts."""
        prompt = self._build_rorschach_encoding_prompt(responses)
        
        result = self.llm.chat(
            user_message=prompt,
            system_prompt=SRAS_ENCODING_SYSTEM_PROMPT,
            config=self.config,
        )
        
        return self._parse_rorschach_encoding(result)
    
    def _build_rorschach_encoding_prompt(self, responses: List[RorschachResponse]) -> str:
        """Build Rorschach encoding prompt."""
        response_text = "\n\n".join([
            f"Card {r.card_number}:\nPerception: {r.perception}\nInquiry: {r.inquiry or 'N/A'}"
            for r in responses
        ])
        
        return f"""Encode the following Rorschach responses according to SRAS variables.

RESPONSES:
{response_text}

Count the following variables across all responses:
- P: Popular responses
- FQo: Ordinary form quality
- FQu: Unusual form quality  
- FQ-: Poor form quality
- WSumCog: Weighted sum of cognitive special scores
- FC: Form-color responses
- CF: Color-form responses
- C: Pure color responses
- C': Achromatic color
- Y: Diffuse shading
- V: Vista
- T: Texture
- M: Human movement
- FM: Animal movement
- m: Inanimate movement
- COP: Cooperative movement
- AG: Aggressive movement
- MOR: Morbid content
- AGC: Aggressive content
- AGM: Aggressive movement
- H: Human content
- M-: Poor form human movement

Output as JSON with variable names as keys and counts as values."""
    
    def _parse_rorschach_encoding(self, output: str) -> Dict[str, int]:
        """Parse Rorschach encoding output."""
        encoding = {}
        try:
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                encoding = json.loads(json_match.group())
                # Ensure all values are integers
                encoding = {k: int(v) for k, v in encoding.items()}
        except (json.JSONDecodeError, ValueError):
            pass
        return encoding
    
    def _calculate_cps(self, enc: Dict[str, int]) -> float:
        """Calculate Cognitive Processing Score."""
        # CPS = 2P + FQo - (FQu + 3FQ- + WSumCog)
        p = enc.get("P", 0)
        fqo = enc.get("FQo", 0)
        fqu = enc.get("FQu", 0)
        fq_minus = enc.get("FQ-", 0)
        wsum_cog = enc.get("WSumCog", 0)
        
        return 2 * p + fqo - (fqu + 3 * fq_minus + wsum_cog)
    
    def _calculate_ars(self, enc: Dict[str, int]) -> float:
        """Calculate Affective Regulation Score."""
        # ARS = 2FC - (CF + 2C + C' + Y + V)
        fc = enc.get("FC", 0)
        cf = enc.get("CF", 0)
        c = enc.get("C", 0)
        c_prime = enc.get("C'", 0)
        y = enc.get("Y", 0)
        v = enc.get("V", 0)
        
        return 2 * fc - (cf + 2 * c + c_prime + y + v)
    
    def _calculate_irs(self, enc: Dict[str, int]) -> float:
        """Calculate Interpersonal Relations Score."""
        # IRS = 3M + 2COP + H - [2(AGC + AGM) + 2MOR + 3M-]
        m = enc.get("M", 0)
        cop = enc.get("COP", 0)
        h = enc.get("H", 0)
        agc = enc.get("AGC", 0)
        agm = enc.get("AGM", 0)
        mor = enc.get("MOR", 0)
        m_minus = enc.get("M-", 0)
        
        return 3 * m + 2 * cop + h - (2 * (agc + agm) + 2 * mor + 3 * m_minus)
    
    def _calculate_scs(self, enc: Dict[str, int]) -> float:
        """Calculate Stress Coping Score."""
        # EA = M + (0.5FC + CF + 1.5C)
        # es = FM + m + Y + T + V + C'
        # SCS = standardize(EA - es)
        m = enc.get("M", 0)
        fc = enc.get("FC", 0)
        cf = enc.get("CF", 0)
        c = enc.get("C", 0)
        fm = enc.get("FM", 0)
        m_inanimate = enc.get("m", 0)
        y = enc.get("Y", 0)
        t = enc.get("T", 0)
        v = enc.get("V", 0)
        c_prime = enc.get("C'", 0)
        
        ea = m + (0.5 * fc + cf + 1.5 * c)
        es = fm + m_inanimate + y + t + v + c_prime
        
        # Simple standardization (can be refined with population norms)
        d = ea - es
        return d  # Raw D-score
    
    def _generate_sras_explanations(
        self,
        enc: Dict[str, int],
        cps: float,
        ars: float,
        irs: float,
        scs: float,
    ) -> Dict[str, str]:
        """Generate explanations for SRAS scores."""
        return {
            "CPS": f"Cognitive Processing Score: {cps:.2f}. Based on perceptual accuracy and conventional responses.",
            "ARS": f"Affective Regulation Score: {ars:.2f}. Reflects emotional modulation and control.",
            "IRS": f"Interpersonal Relations Score: {irs:.2f}. Indicates quality of interpersonal representations.",
            "SCS": f"Stress Coping Score: {scs:.2f}. Balance between resources and psychological burden.",
        }
    
    # ========================================
    # SCT Analysis
    # ========================================
    
    def analyze_sct(
        self,
        responses: List[SCTResponse],
    ) -> SCTScore:
        """
        Analyze SCT responses by domain.
        
        Args:
            responses: List of SCT completions
            
        Returns:
            SCTScore with domain and item scores
        """
        item_scores = {}
        explanations = {}
        
        for response in responses:
            score, explanation = self._score_sct_item(response)
            item_scores[response.stem_id] = score
            explanations[response.stem_id] = explanation
        
        # Calculate domain means
        domain_scores = self._calculate_sct_domain_scores(item_scores)
        
        return SCTScore(
            domain_scores=domain_scores,
            item_scores=item_scores,
            explanations=explanations,
        )
    
    def _score_sct_item(self, response: SCTResponse) -> Tuple[int, str]:
        """Score a single SCT item (0-6 scale)."""
        prompt = f"""Score the following sentence completion on a 0-6 scale:
- 0: Very positive, well-adjusted response
- 3: Neutral, conventional response
- 6: Severely conflicted, maladjusted response

Stem: {response.stem}
Completion: {response.completion}

Output JSON: {{"score": <0-6>, "explanation": "<reason>"}}"""
        
        result = self.llm.chat(
            user_message=prompt,
            system_prompt="You are a clinical psychologist scoring sentence completion tests.",
            config=self.config,
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                data = json.loads(json_match.group())
                return int(data.get("score", 3)), data.get("explanation", "")
        except (json.JSONDecodeError, ValueError):
            pass
        
        return 3, "Unable to parse response"
    
    def _calculate_sct_domain_scores(self, item_scores: Dict[str, int]) -> Dict[str, float]:
        """Calculate mean scores per SCT domain."""
        from ..config import SCT_SUBCONSTRUCT_TO_DOMAIN

        domain_totals: Dict[str, list] = {d: [] for d in ["FA", "CA", "SA", "IR", "ER"]}

        for stem_id, score in item_scores.items():
            # stem_id format: "1.1_01" -> sub-construct "1.1"
            sc_code = stem_id.rsplit("_", 1)[0]
            domain = SCT_SUBCONSTRUCT_TO_DOMAIN.get(sc_code)
            if domain and domain in domain_totals:
                domain_totals[domain].append(score)

        return {
            d: sum(scores) / len(scores) if scores else 3.0
            for d, scores in domain_totals.items()
        }
    
    # ========================================
    # Full Interpretation
    # ========================================
    
    def interpret_all(
        self,
        tat_responses: List[TATResponse],
        rorschach_responses: List[RorschachResponse],
        sct_responses: List[SCTResponse],
    ) -> InterpretationResult:
        """
        Run complete interpretation on all projective test responses.
        
        Returns:
            InterpretationResult with all scores and explanations
        """
        return InterpretationResult(
            tat_scores=self.analyze_tat_batch(tat_responses),
            rorschach_scores=self.analyze_rorschach(rorschach_responses),
            sct_scores=self.analyze_sct(sct_responses),
        )


# ============================================================
# System Prompts
# ============================================================

SCORS_G_SYSTEM_PROMPT = """You are an expert clinical psychologist trained in SCORS-G (Social Cognition and Object Relations Scale - Global) scoring.

The 8 SCORS-G dimensions are:
1. COM (Complexity of Representations): Capacity to see people as complex, multi-dimensional beings
2. AFF (Affective Quality): Valence of relationship expectations (malevolent vs benevolent)
3. EIR (Emotional Investment in Relationships): Depth and investment in relationships
4. EIM (Emotional Investment in Moral Standards): Internalization of moral values
5. SC (Understanding of Social Causality): Ability to understand cause-effect in social interactions
6. AGG (Experience and Management of Aggressive Impulses): How aggression is handled
7. SE (Self-Esteem): Self-regard and self-worth
8. ICS (Identity and Coherence of Self): Sense of stable identity

Score each dimension from 1 (lowest/most pathological) to 7 (highest/healthiest).
Provide specific textual evidence for each score."""

SRAS_ENCODING_SYSTEM_PROMPT = """You are an expert in Rorschach test administration and coding.

Your task is to encode Rorschach responses according to the Simplified Rorschach Analysis System (SRAS).
Count occurrences of each variable across all responses.

Be conservative in your coding - only code features that are clearly present.
If unsure, do not code the variable."""
