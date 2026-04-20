"""
TAT (Thematic Apperception Test) Stimuli Loader

Manages TAT images across three categories (Section 4.1.2, B.3):
- Interpersonal (人际互动): 20 images of human interactions
- Solitary (独处情境): 15 images of single person situations
- Environmental Metaphor (环境隐喻): 5 abstract/symbolic scenes

Per assessment: 8 images in 4:3:1 ratio (Section B.3).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict
from enum import Enum
import random

from ..config import TAT_DIR, TAT_CATEGORY_DIRS, DEFAULT_STIMULI_CONFIG


class TATCategory(Enum):
    """Categories of TAT images."""
    INTERPERSONAL = "interpersonal"
    SOLITARY = "solitary"
    ENVIRONMENTAL = "environmental"


@dataclass
class TATImage:
    """A single TAT image stimulus."""
    id: str
    path: Path
    category: TATCategory
    description: Optional[str] = None

    def exists(self) -> bool:
        return self.path.exists()

    def __str__(self) -> str:
        return f"TAT-{self.id} ({self.category.value})"


@dataclass
class TATStimuli:
    """
    TAT Stimuli collection.

    Manages 40 TAT images:
    - 20 interpersonal scenes (人际互动)
    - 15 solitary scenes (独处情境)
    - 5 environmental metaphor scenes (环境隐喻)
    """

    stimuli_dir: Path = field(default_factory=lambda: TAT_DIR)
    images: List[TATImage] = field(default_factory=list)

    EXPECTED_COUNTS = {
        TATCategory.INTERPERSONAL: 20,
        TATCategory.SOLITARY: 15,
        TATCategory.ENVIRONMENTAL: 5,
    }

    def __post_init__(self):
        if self.stimuli_dir.exists() and not self.images:
            self.load_images()

    def load_images(self) -> None:
        """
        Load TAT images from Chinese-named subdirectories.

        actual structure: gen_stimulis/gen_TAT_figures/
            人际互动/  (interpersonal)
            独处情境/  (solitary)
            环境隐喻/  (environmental)
        """
        self.images = []

        for category in TATCategory:
            chinese_name = TAT_CATEGORY_DIRS.get(category.value, category.value)
            category_dir = self.stimuli_dir / chinese_name
            if not category_dir.exists():
                # fallback to English name
                category_dir = self.stimuli_dir / category.value
            if category_dir.exists():
                image_files = sorted(
                    f for f in category_dir.iterdir()
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
                )
                for idx, img_path in enumerate(image_files, 1):
                    self.images.append(TATImage(
                        id=f"{category.value[:5]}_{idx:02d}",
                        path=img_path,
                        category=category,
                    ))

    def get_by_category(self, category: TATCategory) -> List[TATImage]:
        return [img for img in self.images if img.category == category]

    def get_interpersonal(self) -> List[TATImage]:
        return self.get_by_category(TATCategory.INTERPERSONAL)

    def get_solitary(self) -> List[TATImage]:
        return self.get_by_category(TATCategory.SOLITARY)

    def get_environmental(self) -> List[TATImage]:
        return self.get_by_category(TATCategory.ENVIRONMENTAL)

    def select_for_assessment(
        self,
        num_total: int = 8,
        ratio: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
    ) -> List[TATImage]:
        """
        Select images for a single assessment following paper's 4:3:1 ratio.

        Args:
            num_total: Total images (default 8, Section B.3)
            ratio: Category ratio dict (default 4:3:1)
            seed: Random seed for reproducibility

        Returns:
            Selected TATImage list
        """
        if ratio is None:
            ratio = DEFAULT_STIMULI_CONFIG.tat_ratio

        rng = random.Random(seed)
        selected = []

        for cat_name, count in ratio.items():
            category = TATCategory(cat_name)
            available = self.get_by_category(category)
            if len(available) >= count:
                selected.extend(rng.sample(available, count))
            else:
                selected.extend(available)

        return selected[:num_total]

    def __iter__(self) -> Iterator[TATImage]:
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> TATImage:
        return self.images[idx]

    def validate(self) -> Dict[str, bool]:
        results = {}
        for category, expected_count in self.EXPECTED_COUNTS.items():
            actual = len(self.get_by_category(category))
            results[category.value] = actual >= expected_count
        return results

    def summary(self) -> str:
        lines = ["TAT Stimuli Summary:"]
        for category in TATCategory:
            count = len(self.get_by_category(category))
            expected = self.EXPECTED_COUNTS[category]
            chinese = TAT_CATEGORY_DIRS.get(category.value, "")
            status = "✓" if count >= expected else f"✗ ({count}/{expected})"
            lines.append(f"  {category.value} ({chinese}): {count} images {status}")
        lines.append(f"  Total: {len(self)} images")
        return "\n".join(lines)


# ============================================================
# TAT Prompt Template (Section B.3)
# ============================================================

TAT_PROMPT_TEMPLATE = """Please look at this image carefully and tell a story about it. Include:
1. What is happening in the scene?
2. What events led up to this situation?
3. What are the characters thinking and feeling?
4. What is the possible outcome or ending?

Tell the story naturally, expressing what comes to mind as you look at the image."""
