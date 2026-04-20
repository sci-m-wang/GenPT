"""
SCT (Sentence Completion Test) Stimuli Loader

Loads 97 sentence stems from gen_stimulis/sct_final_filtered.json.
Organized in 4 dimensions, 13 sub-constructs (Section 4.1.2, B.4).

SCT scoring uses 5 domains (FA, CA, SA, IR, ER) mapped from sub-constructs.
Per assessment: 20 stems (4 from each domain).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Iterator
import json
import random

from ..config import SCT_DATA_FILE, SCT_SUBCONSTRUCT_TO_DOMAIN, DEFAULT_STIMULI_CONFIG


@dataclass
class SCTStem:
    """A single sentence completion stem."""
    id: str                  # e.g. "1.1_01"
    stem: str                # Chinese sentence stem
    dimension: str           # dimension name (Chinese)
    sub_construct: str       # sub-construct code e.g. "1.1"
    sub_construct_name: str  # sub-construct name (Chinese)
    domain: str              # mapped scoring domain (FA/CA/SA/IR/ER)

    def __str__(self) -> str:
        return f"SCT-{self.id}: {self.stem}"


@dataclass
class SCTStimuli:
    """
    SCT Stimuli collection.

    Loads stems from sct_final_filtered.json.
    Structure: {"SCT_WB_C": [dimension_objects]}
    Each dimension has sub-constructs, each containing items.
    """

    data_file: Path = field(default_factory=lambda: SCT_DATA_FILE)
    stems: List[SCTStem] = field(default_factory=list)
    _by_domain: Dict[str, List[SCTStem]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.data_file.exists() and not self.stems:
            self.load_stems()

    def load_stems(self) -> None:
        """
        Load stems from JSON file.

        JSON structure:
        {
          "SCT_WB_C": [
            {
              "dimension": "关系福祉",
              "sub_constructs": [
                {
                  "sub_construct": "1.1 亲密关系满意度",
                  "items": ["我觉得我的母亲很少……", ...]
                }, ...
              ]
            }, ...
          ]
        }
        """
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stems = []
        sct_obj = data.get("SCT_WB_C", data)

        # Handle nested dict structure: SCT_WB_C is a dict with 'dimensions' list
        if isinstance(sct_obj, dict):
            dimensions = sct_obj.get("dimensions", [])
        else:
            dimensions = sct_obj

        for dim_obj in dimensions:
            dim_name = dim_obj.get("dimension_name", dim_obj.get("dimension", ""))

            # Sub-constructs may be under 'secondary_constructs' or 'sub_constructs'
            sub_constructs = (
                dim_obj.get("secondary_constructs", [])
                or dim_obj.get("sub_constructs", [])
            )

            for sc_obj in sub_constructs:
                # construct_id / construct_name or sub_construct
                sc_code = sc_obj.get("construct_id", "")
                sc_name = sc_obj.get("construct_name", "")

                if not sc_code:
                    # Fallback: parse from "1.1 亲密关系满意度" format
                    sc_raw = sc_obj.get("sub_construct", "")
                    parts = sc_raw.split(" ", 1)
                    sc_code = parts[0] if parts else sc_raw
                    sc_name = parts[1] if len(parts) > 1 else ""

                # Map sub-construct to scoring domain
                domain = SCT_SUBCONSTRUCT_TO_DOMAIN.get(sc_code, "OTHER")

                items = sc_obj.get("items", [])
                for item_idx, stem_text in enumerate(items, 1):
                    # Strip leading number prefix like "1. " or "45. "
                    clean = stem_text.strip()
                    import re as _re
                    clean = _re.sub(r'^\d+\.\s*', '', clean)

                    stem_id = f"{sc_code}_{item_idx:02d}"
                    self.stems.append(SCTStem(
                        id=stem_id,
                        stem=clean,
                        dimension=dim_name,
                        sub_construct=sc_code,
                        sub_construct_name=sc_name,
                        domain=domain,
                    ))

        # Build domain index
        self._build_domain_index()

    def _build_domain_index(self) -> None:
        """Build lookup from scoring domain to stems."""
        self._by_domain = {}
        for stem in self.stems:
            self._by_domain.setdefault(stem.domain, []).append(stem)

    def get_by_domain(self, domain: str) -> List[SCTStem]:
        """Get stems by scoring domain (FA, CA, SA, IR, ER)."""
        return self._by_domain.get(domain, [])

    def get_by_dimension(self, dimension: str) -> List[SCTStem]:
        """Get stems by original Chinese dimension."""
        return [s for s in self.stems if s.dimension == dimension]

    def get_by_subconstruct(self, sc_code: str) -> List[SCTStem]:
        """Get stems by sub-construct code (e.g., '1.1')."""
        return [s for s in self.stems if s.sub_construct == sc_code]

    def get_domains(self) -> List[str]:
        """Return all scoring domains present."""
        return sorted(set(s.domain for s in self.stems))

    def select_for_assessment(
        self,
        num_total: int = 20,
        per_domain: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[SCTStem]:
        """
        Select stems for a single assessment (Section B.4).

        Paper: 20 stems, 4 from each of 5 scoring domains.

        Args:
            num_total: Total stems (default 20)
            per_domain: Per-domain count (default: num_total / num_domains)
            seed: Random seed for reproducibility

        Returns:
            Selected SCTStem list
        """
        rng = random.Random(seed)
        domains = self.get_domains()

        if per_domain is None:
            per_domain = DEFAULT_STIMULI_CONFIG.sct_per_domain

        selected = []
        for domain in domains:
            available = self.get_by_domain(domain)
            if len(available) >= per_domain:
                selected.extend(rng.sample(available, per_domain))
            else:
                selected.extend(available)

        # If we have more than needed, trim
        if len(selected) > num_total:
            selected = rng.sample(selected, num_total)

        return selected

    def __iter__(self) -> Iterator[SCTStem]:
        return iter(self.stems)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> SCTStem:
        return self.stems[idx]

    def validate(self) -> Dict[str, bool]:
        domains = ["FA", "CA", "SA", "IR", "ER"]
        return {
            d: len(self.get_by_domain(d)) > 0
            for d in domains
        }

    def summary(self) -> str:
        lines = ["SCT Stimuli Summary:"]
        for domain in sorted(self._by_domain.keys()):
            stems = self._by_domain[domain]
            lines.append(f"  Domain {domain}: {len(stems)} stems")
        lines.append(f"  Total: {len(self)} stems")
        return "\n".join(lines)


# ============================================================
# SCT Prompt Template (Section B.4)
# ============================================================

SCT_PROMPT_TEMPLATE = """请完成以下句子。没有对错之分，请用第一反应来回答。

{stem}"""
