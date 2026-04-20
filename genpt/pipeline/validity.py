"""
Validity-test conditions: social-desirability framings + long-context injection.

Used by `scripts/generate_behaviors.py` (projective validity tests) and
`scripts/evaluate_questionnaire.py` (questionnaire validity tests) to
re-run each evaluation under controlled perturbations of the examinee setup.

Design principles:
- SDB scenarios are **single extra system messages** inserted between the
  persona definition and the elicitation turn. They describe a *situation*
  (job screening / clinical intake) without telling the subject to fake good
  or fake bad. Any social-desirability effect has to emerge from the context,
  not from an explicit instruction. The two scenarios are symmetric in length
  and register.
- Long-context injects 10 real prior turns (counselor/patient for AnnaAgent,
  interviewer Q&A for CharacterRAG) as multi-turn `messages` before the
  elicitation turn. The examinee inherits that conversational state, so TAT /
  SCT / Rorschach / questionnaire answers are conditioned on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from genpt.pipeline.examinee import Message


# ---------------------------------------------------------------------------
# SDB scenarios
# ---------------------------------------------------------------------------
# Both scenarios share structure: "You are participating in <context>. A
# <reviewer> will look at your responses as part of <purpose>. Answer as you
# normally would." The wording avoids words like "healthy", "troubled",
# "well", "struggling", "good impression", "honest", etc. so the prompt itself
# carries no valence.

SDB_JOB = (
    "Situational context for this session: you are participating in a "
    "psychological screening as part of a job application process. The "
    "materials will be reviewed by the human-resources department of the "
    "company you applied to, alongside your CV and interview notes. The "
    "results will inform their hiring decision. Continue with the tasks "
    "you are given, responding as you normally would."
)

SDB_CLINICAL = (
    "Situational context for this session: you are participating in a "
    "psychological assessment at a mental-health clinic, after you recently "
    "contacted them about ongoing difficulties. The materials will be "
    "reviewed by the clinical team who will plan your follow-up sessions "
    "and, if applicable, discuss treatment options. Continue with the tasks "
    "you are given, responding as you normally would."
)


def sdb_system_message(scenario: str) -> str:
    scenario = scenario.lower()
    if scenario in ("job", "sdb_job", "sdb-job", "interview"):
        return SDB_JOB
    if scenario in ("clinical", "clin", "sdb_clinical", "sdb-clinical", "clinic"):
        return SDB_CLINICAL
    raise ValueError(f"Unknown SDB scenario: {scenario}")


# ---------------------------------------------------------------------------
# Long-context prefix history
# ---------------------------------------------------------------------------

_D4_S_CACHE: Optional[Dict[str, List[Dict]]] = None
_CRAG_QA_CACHE: Dict[str, List[Tuple[str, str]]] = {}


def _load_d4s_index(repo_root: Path) -> Dict[str, List[Dict]]:
    """Index D4_S.json by id → conversation list."""
    global _D4_S_CACHE
    if _D4_S_CACHE is not None:
        return _D4_S_CACHE
    path = repo_root / "characters" / "AnnaAgent" / "D4_S.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _D4_S_CACHE = {item["id"]: item.get("conversation", []) for item in data}
    return _D4_S_CACHE


def annaagent_prefix_history(
    repo_root: Path, source_id: str, num_turns: int = 10
) -> List[Message]:
    """
    Return the first `num_turns` turns of the AnnaAgent patient's counselor
    conversation as Message objects.

    - counselor speaking → role="user" (the examinee, who IS the patient,
      responded to the counselor)
    - seeker/patient speaking → role="assistant" (those are the examinee's
      past utterances)

    We stop at `num_turns` total messages, preserving order. If the conversation
    is shorter than `num_turns`, return whatever is available. If the first
    utterance is from the patient (seeker), we pad with a short counselor
    greeting so the history begins with a user turn (required by most chat
    templates including Qwen3).
    """
    index = _load_d4s_index(repo_root)
    conv = index.get(source_id, [])
    out: List[Message] = []
    for entry in conv[:num_turns]:
        role_raw = (entry.get("role") or "").lower()
        content = entry.get("content", "").strip()
        if not content:
            continue
        if role_raw in ("counselor", "therapist", "doctor", "assistant_counselor"):
            role = "user"
        elif role_raw in ("seeker", "patient", "client", "user_patient"):
            role = "assistant"
        else:
            # unknown role — skip
            continue
        out.append(Message(role=role, content=content))

    # Ensure history starts with user turn (counselor).
    if out and out[0].role != "user":
        out.insert(0, Message(role="user", content="Hello, let's begin."))
        out = out[:num_turns]
    return out


def _load_crag_qa(repo_root: Path, source_key: str) -> List[Tuple[str, str]]:
    """Load (question, answer) pairs from the CharacterRAG xlsx for a char."""
    if source_key in _CRAG_QA_CACHE:
        return _CRAG_QA_CACHE[source_key]
    import pandas as pd  # local import; openpyxl only installed lazily

    char_dir = repo_root / "characters" / "CharacterRAG" / source_key
    xlsx = char_dir / f"{source_key}_en.xlsx"
    if not xlsx.exists():
        xlsx = char_dir / f"{source_key}.xlsx"
    if not xlsx.exists():
        _CRAG_QA_CACHE[source_key] = []
        return []
    df = pd.read_excel(xlsx)
    # Normalise column names (the CRAG file has a typo 'attirbute').
    cols = {c.lower().strip(): c for c in df.columns}
    q_col = cols.get("question")
    a_col = cols.get("answer")
    if q_col is None or a_col is None:
        _CRAG_QA_CACHE[source_key] = []
        return []
    pairs: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        q = str(row[q_col]).strip()
        a = str(row[a_col]).strip()
        if q and a and q.lower() != "nan" and a.lower() != "nan":
            pairs.append((q, a))
    _CRAG_QA_CACHE[source_key] = pairs
    return pairs


def characterrag_prefix_history(
    repo_root: Path, source_key: str, num_pairs: int = 10
) -> List[Message]:
    """
    Return the first `num_pairs` Q&A pairs from the CRAG xlsx as alternating
    user/assistant messages (question=user, answer=assistant).
    """
    pairs = _load_crag_qa(repo_root, source_key)
    out: List[Message] = []
    for q, a in pairs[:num_pairs]:
        out.append(Message(role="user", content=q))
        out.append(Message(role="assistant", content=a))
    return out


def build_condition(
    condition: str,
    repo_root: Path,
    source_type: str,
    source_key: str,
    num_turns: int = 10,
) -> Tuple[List[str], List[Message]]:
    """
    Return (extra_system_messages, prefix_history) for the requested condition.

    Conditions:
        - "baseline" / None: empty perturbation (recovers original behaviour).
        - "sdb_job": append SDB_JOB as extra system message.
        - "sdb_clinical": append SDB_CLINICAL as extra system message.
        - "longctx": inject 10 prior-context messages appropriate for the
          source (AnnaAgent → D4_S counselor dialogue; CharacterRAG → xlsx QA).
    """
    condition = (condition or "baseline").lower()
    if condition in ("baseline", "none", ""):
        return [], []
    if condition in ("sdb_job", "sdb-job", "job"):
        return [SDB_JOB], []
    if condition in ("sdb_clinical", "sdb-clinical", "clinical", "clin"):
        return [SDB_CLINICAL], []
    if condition in ("longctx", "long_context", "long-context", "long"):
        if source_type == "annaagent":
            history = annaagent_prefix_history(repo_root, source_key, num_turns)
        elif source_type == "characterrag":
            history = characterrag_prefix_history(repo_root, source_key, num_turns)
        else:
            history = []
        return [], history
    raise ValueError(f"Unknown condition: {condition}")


__all__ = [
    "SDB_JOB",
    "SDB_CLINICAL",
    "sdb_system_message",
    "annaagent_prefix_history",
    "characterrag_prefix_history",
    "build_condition",
]
