"""
SRAS (Simplified Rorschach Analysis System) Scoring

Implements a simplified Rorschach encoding and scoring system 
adapted for LLM-based Examinees, focusing on content extractable 
from utterance records.

Domain Scores:
- CPS: Cognitive Processing Score
- ARS: Affective Regulation Score
- IRS: Interpersonal Relations Score
- SCS: Stress Coping Score
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SRASVariable(Enum):
    """SRAS encoding variables."""
    # Form Quality
    P = "Popular"
    FQo = "Ordinary Form Quality"
    FQu = "Unusual Form Quality"
    FQ_MINUS = "Poor Form Quality"
    FQ_NONE = "No Form"
    
    # Color
    FC = "Form-Color"
    CF = "Color-Form"
    C = "Pure Color"
    C_PRIME = "Achromatic Color"
    
    # Shading
    Y = "Diffuse Shading"
    T = "Texture"
    V = "Vista"
    
    # Movement
    M = "Human Movement"
    FM = "Animal Movement"
    m = "Inanimate Movement"
    M_MINUS = "Poor Form Human Movement"
    
    # Content
    H = "Human Content"
    Hd = "Human Detail"
    A = "Animal Content"
    Ad = "Animal Detail"
    
    # Special Scores - Cognitive
    DV = "Deviant Verbalization"
    DR = "Deviant Response"
    INC = "Incongruous Combination"
    FAB = "Fabulized Combination"
    ALOG = "Autistic Logic"
    CONTAM = "Contamination"
    
    # Special Scores - Interpersonal
    COP = "Cooperative Movement"
    AG = "Aggressive Movement"
    MOR = "Morbid Content"
    AGC = "Aggressive Content"
    AGM = "Aggressive Movement"


# Cognitive special score weights for WSumCog
COGNITIVE_WEIGHTS = {
    SRASVariable.DV: 1,
    SRASVariable.DR: 2,
    SRASVariable.INC: 2,
    SRASVariable.FAB: 3,
    SRASVariable.ALOG: 4,
    SRASVariable.CONTAM: 7,
}


@dataclass
class SRASEncoding:
    """Complete SRAS encoding for a Rorschach protocol."""
    variables: Dict[str, int] = field(default_factory=dict)
    card_encodings: List[Dict] = field(default_factory=list)
    
    def get(self, var: str, default: int = 0) -> int:
        """Get count for a variable."""
        return self.variables.get(var, default)
    
    def set(self, var: str, count: int) -> None:
        """Set count for a variable."""
        self.variables[var] = count
    
    def increment(self, var: str, amount: int = 1) -> None:
        """Increment a variable count."""
        self.variables[var] = self.variables.get(var, 0) + amount
    
    def calculate_wsum_cog(self) -> int:
        """Calculate weighted sum of cognitive special scores."""
        total = 0
        for var, weight in COGNITIVE_WEIGHTS.items():
            total += self.get(var.name, 0) * weight
        return total
    
    def total_responses(self) -> int:
        """Estimate total number of responses."""
        return len(self.card_encodings) if self.card_encodings else 10


@dataclass
class SRASDomainScores:
    """Domain scores calculated from SRAS encoding."""
    cps: float = 0.0  # Cognitive Processing Score
    ars: float = 0.0  # Affective Regulation Score
    irs: float = 0.0  # Interpersonal Relations Score
    scs: float = 0.0  # Stress Coping Score
    
    # Intermediate values
    ea: float = 0.0   # Experiential Availability
    es: float = 0.0   # Experiential Stimulation
    
    def to_vector(self) -> List[float]:
        return [self.cps, self.ars, self.irs, self.scs]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "CPS": self.cps,
            "ARS": self.ars,
            "IRS": self.irs,
            "SCS": self.scs,
        }


class SRASEncoder:
    """
    Encodes Rorschach responses into SRAS variables.
    
    Provides structured guidance for encoding process.
    """
    
    VARIABLES = list(SRASVariable)
    
    @staticmethod
    def get_encoding_guidelines() -> Dict[str, str]:
        """Get guidelines for encoding each variable."""
        return {
            "P": "Common, frequently given responses to specific card areas",
            "FQo": "Response uses form appropriately for the blot area",
            "FQu": "Response is unusual but form is used adequately",
            "FQ-": "Response misuses form; doesn't fit the blot area",
            "FC": "Color integrated with form dominance",
            "CF": "Color dominates with some form",
            "C": "Pure color response with no form",
            "C'": "Achromatic color (black, white, gray) used as color",
            "Y": "Diffuse shading (undifferentiated)",
            "T": "Tactile/texture shading",
            "V": "Vista/depth shading",
            "M": "Human movement (action attribution)",
            "FM": "Animal movement",
            "m": "Inanimate movement (explosion, falling, etc.)",
            "H": "Whole human figure",
            "COP": "Two or more figures in cooperative/positive interaction",
            "AG": "Aggressive movement or action",
            "MOR": "Morbid content (death, damage, decay)",
        }
    
    @staticmethod
    def create_empty_encoding() -> SRASEncoding:
        """Create an empty encoding structure."""
        return SRASEncoding(
            variables={var.name: 0 for var in SRASVariable}
        )


class SRASScorer:
    """
    Calculates SRAS domain scores from encodings.
    
    Implements the formulas from the GenPT paper:
    - CPS = 2P + FQo - (FQu + 3FQ- + WSumCog)
    - ARS = 2FC - (CF + 2C + C' + Y + V)
    - IRS = 3M + 2COP + H - [2(AGC + AGM) + 2MOR + 3M-]
    - SCS = standardize(EA - es) where EA = M + (0.5FC + CF + 1.5C), es = FM + m + Y + T + V + C'
    """
    
    @staticmethod
    def calculate_cps(enc: SRASEncoding) -> float:
        """
        Calculate Cognitive Processing Score.
        
        CPS reflects clarity of thought and reality testing.
        Increases with conventional, accurate perceptions.
        Decreases with distorted or illogical responses.
        """
        p = enc.get("P")
        fqo = enc.get("FQo")
        fqu = enc.get("FQu")
        fq_minus = enc.get("FQ-") or enc.get("FQ_MINUS")
        wsum_cog = enc.calculate_wsum_cog()
        
        return 2 * p + fqo - (fqu + 3 * fq_minus + wsum_cog)
    
    @staticmethod
    def calculate_ars(enc: SRASEncoding) -> float:
        """
        Calculate Affective Regulation Score.
        
        ARS measures emotional modulation.
        Rewards controlled emotional responses.
        Penalizes unregulated or painful affect.
        """
        fc = enc.get("FC")
        cf = enc.get("CF")
        c = enc.get("C")
        c_prime = enc.get("C'") or enc.get("C_PRIME")
        y = enc.get("Y")
        v = enc.get("V")
        
        return 2 * fc - (cf + 2 * c + c_prime + y + v)
    
    @staticmethod
    def calculate_irs(enc: SRASEncoding) -> float:
        """
        Calculate Interpersonal Relations Score.
        
        IRS captures how the Examinee perceives people and relationships.
        Positive indicators: human movement, cooperation, human content.
        Negative indicators: aggression, morbid content.
        """
        m = enc.get("M")
        cop = enc.get("COP")
        h = enc.get("H")
        agc = enc.get("AGC")
        agm = enc.get("AGM")
        mor = enc.get("MOR")
        m_minus = enc.get("M-") or enc.get("M_MINUS")
        
        return 3 * m + 2 * cop + h - (2 * (agc + agm) + 2 * mor + 3 * m_minus)
    
    @staticmethod
    def calculate_ea(enc: SRASEncoding) -> float:
        """Calculate Experiential Availability (EA)."""
        m = enc.get("M")
        fc = enc.get("FC")
        cf = enc.get("CF")
        c = enc.get("C")
        
        return m + (0.5 * fc + cf + 1.5 * c)
    
    @staticmethod
    def calculate_es(enc: SRASEncoding) -> float:
        """Calculate Experiential Stimulation (es)."""
        fm = enc.get("FM")
        m_inanimate = enc.get("m")
        y = enc.get("Y")
        t = enc.get("T")
        v = enc.get("V")
        c_prime = enc.get("C'") or enc.get("C_PRIME")
        
        return fm + m_inanimate + y + t + v + c_prime
    
    @staticmethod
    def calculate_scs(enc: SRASEncoding) -> float:
        """
        Calculate Stress Coping Score.
        
        SCS reflects the balance between internal resources and psychological burden.
        Positive: more resources available (EA) than demands (es).
        """
        ea = SRASScorer.calculate_ea(enc)
        es = SRASScorer.calculate_es(enc)
        
        # D score (difference)
        d = ea - es
        
        # Simple standardization to approximate T-score
        # In practice, this would use population norms
        return d  # Raw D-score
    
    @classmethod
    def calculate_all(cls, enc: SRASEncoding) -> SRASDomainScores:
        """Calculate all domain scores from encoding."""
        ea = cls.calculate_ea(enc)
        es = cls.calculate_es(enc)
        
        return SRASDomainScores(
            cps=cls.calculate_cps(enc),
            ars=cls.calculate_ars(enc),
            irs=cls.calculate_irs(enc),
            scs=cls.calculate_scs(enc),
            ea=ea,
            es=es,
        )
    
    @staticmethod
    def interpret_scores(scores: SRASDomainScores) -> Dict[str, str]:
        """Provide interpretations for domain scores."""
        interpretations = {}
        
        # CPS interpretation
        if scores.cps > 5:
            interpretations["CPS"] = "Good cognitive processing; reality testing intact"
        elif scores.cps > 0:
            interpretations["CPS"] = "Adequate cognitive processing"
        else:
            interpretations["CPS"] = "Cognitive processing difficulties; may have perceptual distortions"
        
        # ARS interpretation
        if scores.ars > 2:
            interpretations["ARS"] = "Well-regulated affect; emotional control"
        elif scores.ars > -2:
            interpretations["ARS"] = "Moderate affective regulation"
        else:
            interpretations["ARS"] = "Affective dysregulation; emotional difficulties"
        
        # IRS interpretation
        if scores.irs > 5:
            interpretations["IRS"] = "Positive interpersonal representations"
        elif scores.irs > 0:
            interpretations["IRS"] = "Mixed interpersonal functioning"
        else:
            interpretations["IRS"] = "Interpersonal difficulties; negative expectations"
        
        # SCS interpretation
        if scores.scs > 0:
            interpretations["SCS"] = "Resources exceed demands; good stress tolerance"
        else:
            interpretations["SCS"] = "Demands may exceed resources; stress vulnerability"
        
        return interpretations
