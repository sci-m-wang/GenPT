"""
Rorschach Inkblot Test Stimuli Loader

Manages 10 Rorschach cards (I-X), administered sequentially
in multi-turn dialogue (Section 4.1.2, B.2).

Structure: gen_stimulis/gen_Rorschach_figures/{I,II,...,X}/*.png
Some cards have multiple generated variants; one is selected per assessment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator, Dict
import random

from ..config import RORSCHACH_DIR, RORSCHACH_CARD_DIRS


@dataclass
class RorschachCard:
    """A single Rorschach card stimulus."""
    card_number: int          # 1-10
    roman_numeral: str        # "I"-"X"
    path: Path
    variants: List[Path] = field(default_factory=list)

    def exists(self) -> bool:
        return self.path.exists()

    def __str__(self) -> str:
        return f"Rorschach Card {self.roman_numeral}"


@dataclass
class RorschachStimuli:
    """
    Rorschach Stimuli collection.

    10 inkblot cards administered sequentially (I→X).
    Multi-turn dialogue: response phase + inquiry phase per card.
    """

    stimuli_dir: Path = field(default_factory=lambda: RORSCHACH_DIR)
    cards: List[RorschachCard] = field(default_factory=list)

    NUM_CARDS = 10

    def __post_init__(self):
        if self.stimuli_dir.exists() and not self.cards:
            self.load_cards()

    def load_cards(self) -> None:
        """
        Load Rorschach cards from Roman numeral subdirectories.

        Structure: gen_Rorschach_figures/
            I/   (may contain 1+ .png variants)
            II/
            ...
            X/
        """
        self.cards = []

        for idx, roman in enumerate(RORSCHACH_CARD_DIRS, 1):
            card_dir = self.stimuli_dir / roman
            if not card_dir.exists():
                continue

            image_files = sorted(
                f for f in card_dir.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
            )
            if not image_files:
                continue

            # Use the first variant as default, store all as variants
            self.cards.append(RorschachCard(
                card_number=idx,
                roman_numeral=roman,
                path=image_files[0],
                variants=image_files,
            ))

        # Sort by card number
        self.cards.sort(key=lambda c: c.card_number)

    def select_variants(self, seed: Optional[int] = None) -> List[RorschachCard]:
        """
        Select one variant per card for assessment.

        If a card has multiple generated variants, randomly pick one.
        Returns list of 10 cards in order I-X.
        """
        rng = random.Random(seed)
        selected = []

        for card in self.cards:
            if len(card.variants) > 1:
                chosen = rng.choice(card.variants)
                selected.append(RorschachCard(
                    card_number=card.card_number,
                    roman_numeral=card.roman_numeral,
                    path=chosen,
                    variants=card.variants,
                ))
            else:
                selected.append(card)

        return selected

    def get_card(self, number: int) -> Optional[RorschachCard]:
        """Get card by number (1-indexed)."""
        for card in self.cards:
            if card.card_number == number:
                return card
        return None

    def __iter__(self) -> Iterator[RorschachCard]:
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, idx: int) -> RorschachCard:
        return self.cards[idx]

    def validate(self) -> Dict[str, bool]:
        found = {c.card_number for c in self.cards}
        return {
            roman: (i in found)
            for i, roman in enumerate(RORSCHACH_CARD_DIRS, 1)
        }

    def summary(self) -> str:
        lines = ["Rorschach Stimuli Summary:"]
        for card in self.cards:
            lines.append(
                f"  Card {card.roman_numeral}: {card.path.name} "
                f"({len(card.variants)} variant(s))"
            )
        lines.append(f"  Total: {len(self)}/{self.NUM_CARDS} cards loaded")
        return "\n".join(lines)


# ============================================================
# Rorschach Prompt Templates (Section B.2)
# ============================================================

# Free Association Phase (per card)
RORSCHACH_FREE_ASSOCIATION_PROMPT = """Please look at this inkblot image carefully. What might this be? What does it look like to you?

Tell me everything you see. There are no right or wrong answers."""

# Inquiry Phase (follow-up after each response)
RORSCHACH_INQUIRY_PROMPT = """Thank you for your response. I'd like to ask a few follow-up questions:
1. Where in the image did you see that?
2. What about the image made it look like that to you?
3. Is there anything else you notice?"""
