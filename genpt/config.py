"""
GenPT Configuration Module

Contains global settings, constants, and paths for the GenPT framework.
Aligned with the GenPT paper (6267_GenPT_Beyond_Self_Report).
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================
# Path Configuration
# ============================================================

ROOT_DIR = Path(__file__).parent.parent
GEN_STIMULI_DIR = ROOT_DIR / "gen_stimulis"
DATA_DIR = ROOT_DIR / "data"
CHARACTERS_DIR = ROOT_DIR / "characters"
QUESTIONNAIRES_DIR = ROOT_DIR / "questionnaires"

# Stimuli subdirectories (matching actual directory structure)
TAT_DIR = GEN_STIMULI_DIR / "gen_TAT_figures"
RORSCHACH_DIR = GEN_STIMULI_DIR / "gen_Rorschach_figures"
SCT_DATA_FILE = GEN_STIMULI_DIR / "sct_final_filtered.json"

# Character data
CHARACTERRAG_DIR = CHARACTERS_DIR / "CharacterRAG"
ANNAAGENT_DIR = CHARACTERS_DIR / "AnnaAgent"
PDB_LABELS_PATH = CHARACTERRAG_DIR / "pdb_labels.json"
ANNAAGENT_DATA_PATH = ANNAAGENT_DIR / "D4_prompts_with_labels.json"


# ============================================================
# Model Configuration
# ============================================================

import os

MODELS_DIR = Path(os.environ.get("GENPT_MODELS_DIR", "/home/aiscuser/models"))


@dataclass
class ModelConfig:
    """Configuration for LLM models."""

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_path: Optional[str] = None  # local path override
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None  # None = no limit, model generates until EOS
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"
    use_vllm: bool = False
    tensor_parallel_size: int = 1
    # Whether the model is multimodal (VL)
    is_multimodal: bool = False
    # Whether to enable thinking mode (Qwen3)
    enable_thinking: bool = False


# Examinee: Qwen3-VL-8B-Instruct (multimodal for TAT images + Rorschach inkblots)
EXAMINEE_MODEL_CONFIG = ModelConfig(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    model_path=str(MODELS_DIR / "Qwen" / "Qwen3-VL-8B-Instruct"),
    is_multimodal=True,
    enable_thinking=False,
)

# Interpreter & Diagnostician: Qwen3-8B (text-only, thinking enabled)
INTERPRETER_MODEL_CONFIG = ModelConfig(
    model_name="Qwen/Qwen3-8B",
    model_path=str(MODELS_DIR / "Qwen" / "Qwen3-8B"),
    is_multimodal=False,
    enable_thinking=True,
)

DEFAULT_MODEL_CONFIG = EXAMINEE_MODEL_CONFIG


# ============================================================
# SCORS-G Dimensions (for TAT analysis, Section 4.2.1)
# ============================================================

SCORS_G_DIMENSIONS = [
    "COM",  # Complexity of Representations of People
    "AFF",  # Affective Quality of Representations
    "EIR",  # Emotional Investment in Relationships
    "EIM",  # Emotional Investment in Moral Standards
    "SC",  # Understanding of Social Causality
    "AGG",  # Experience and Management of Aggressive Impulses
    "SE",  # Self-Esteem
    "ICS",  # Identity and Coherence of Self
]

SCORS_G_SCORE_RANGE = (1, 7)


# ============================================================
# SRAS Variables (for Rorschach analysis, Section 4.2.2)
# ============================================================

SRAS_FQ_CODES = ["FQo", "FQu", "FQ-", "FQnone"]
SRAS_CF_CODES = ["FC", "CF", "C"]
SRAS_COG_CODES = ["DV", "DR", "INC", "FAB", "ALOG", "CONTAM"]
SRAS_CONTENT_CODES = [
    "H",
    "Hd",
    "A",
    "Ad",
    "An",
    "Art",
    "Bl",
    "Cg",
    "Cl",
    "Ex",
    "Fi",
    "Fd",
    "Ge",
    "Hh",
    "Ls",
    "Na",
    "Sc",
    "Sx",
    "Xy",
]
SRAS_MOVEMENT_CODES = ["M", "FM", "m"]
SRAS_INTERP_CODES = ["COP", "AG", "MOR", "AGC", "AGM"]
SRAS_SHADING_CODES = ["Y", "T", "V", "C'"]
SRAS_DOMAIN_SCORES = ["CPS", "ARS", "IRS", "SCS"]


# ============================================================
# SCT Domains (Section 4.2.3, Equation 9)
# ============================================================

SCT_DOMAINS = {
    "FA": "Family Adjustment",
    "CA": "Career Adjustment",
    "SA": "Self-Attitudes",
    "IR": "Interpersonal Relationships",
    "ER": "Emotion Regulation",
}

SCT_SCORE_RANGE = (0, 6)

# Mapping from Chinese sub-construct IDs to the 5 scoring domains
SCT_SUBCONSTRUCT_TO_DOMAIN = {
    "1.1": "FA",  # 对母亲形象的态度
    "1.2": "FA",  # 对父亲形象的态度
    "1.3": "FA",  # 家庭单位与和谐度
    "1.4": "IR",  # 同伴与社会关系
    "2.1": "SA",  # 自主与自我导向
    "2.2": "CA",  # 环境掌控与胜任力
    "2.3": "SA",  # 个人成长与开放性
    "2.4": "CA",  # 成就与抱负
    "3.1": "SA",  # 生活目标与意义
    "3.2": "SA",  # 自我接纳与过往视角
    "3.3": "ER",  # 情绪体验与调适
    "4.1": "CA",  # "内卷"体验
    "4.2": "ER",  # 对"躺平"的态度
    "4.3": "CA",  # 未来与职业定向
}


# ============================================================
# Rorschach Explanation Vector (Section A.2)
# E(ro) = {F, DQ, R, P, C, M, COG, AFF}
# ============================================================

RORSCHACH_EXPLANATION_DIMS = ["F", "DQ", "R", "P", "C", "M", "COG", "AFF"]


# ============================================================
# Target Tasks (Section 4.3, Equations 10-11)
# ============================================================


@dataclass
class TaskConfig:
    """Configuration for assessment tasks."""

    name: str
    output_type: str
    dimensions: List[str]
    score_range: tuple


BIG_FIVE_TASK = TaskConfig(
    name="Big Five",
    output_type="categorical",
    dimensions=[
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ],
    score_range=(1, 5),
)

MBTI_TASK = TaskConfig(
    name="MBTI",
    output_type="continuous",
    dimensions=["E-I", "S-N", "T-F", "J-P"],
    score_range=(0.0, 1.0),
)

DEPRESSION_TASK = TaskConfig(
    name="Depression Risk",
    output_type="categorical",
    dimensions=["depression_level"],
    score_range=(0, 3),
)

SUICIDE_TASK = TaskConfig(
    name="Suicide Risk",
    output_type="categorical",
    dimensions=["suicide_ideation"],
    score_range=(0, 3),
)


# ============================================================
# Assessment Stimuli Config (Section B.3)
# ============================================================


@dataclass
class AssessmentStimuliConfig:
    """Paper-specified stimuli counts per assessment."""

    # TAT: 8 images, 4:3:1 ratio (Section B.3)
    num_tat_images: int = 8
    tat_ratio: Dict[str, int] = field(
        default_factory=lambda: {"interpersonal": 4, "solitary": 3, "environmental": 1}
    )
    # Rorschach: all 10 cards
    num_rorschach_cards: int = 10
    # SCT/RISB: 20 sentence stems, 4 per domain
    num_sct_stems: int = 20
    sct_per_domain: int = 4


DEFAULT_STIMULI_CONFIG = AssessmentStimuliConfig()


# ============================================================
# Experiment Settings (Section 5)
# ============================================================


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    random_seed: int = 42
    # Test-retest (Section 5.3.1): 10 conversation turns gap
    test_retest_gap_turns: int = 10
    # Multiple assessment (Section 5.3.2): 5 repetitions
    num_repeated_measures: int = 5
    # Output paths
    results_dir: Path = field(default_factory=lambda: DATA_DIR / "results")
    logs_dir: Path = field(default_factory=lambda: DATA_DIR / "logs")

    def ensure_dirs(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


# ============================================================
# Social Desirability Framing Prompts (Section 5.3.3)
# ============================================================

SOCIAL_DESIRABILITY_PROMPTS = {
    "neutral": "",
    "positive": (
        "You are currently in the midst of a crucial job interview... "
        "Please show your best self."
    ),
    "negative": (
        "You are now in a completely secure and confidential counselling room... "
        "Answer honestly without concern for evaluation."
    ),
}


# ============================================================
# Directory Mappings for Stimuli
# ============================================================

TAT_CATEGORY_DIRS = {
    "interpersonal": "人际互动",
    "solitary": "独处情境",
    "environmental": "环境隐喻",
}

RORSCHACH_CARD_DIRS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
