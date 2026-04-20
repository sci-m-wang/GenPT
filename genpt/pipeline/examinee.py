"""
Examinee Module (Stage 1: Response Generation)

The Examinee represents the LLM under assessment, instantiated with a specific persona.
It generates free-form responses to projective test stimuli.

Supports two character data sources:
- AnnaAgent: Direct system prompts from JSON files
- CharacterRAG: Structured profile attributes from TXT files (Beliefs, Traits, Speech Style)

Equation: R = X(T) = {r_1, ..., r_n}
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import json
import re

from ..llm.base import BaseLLM, Message, GenerationConfig
from ..stimuli.tat import TATImage, TAT_PROMPT_TEMPLATE
from ..stimuli.rorschach import RorschachCard, RORSCHACH_FREE_ASSOCIATION_PROMPT, RORSCHACH_INQUIRY_PROMPT
from ..stimuli.sct import SCTStem, SCT_PROMPT_TEMPLATE


# CharacterRAG prompt template (AMADEUS-style attributes + knowledge)
CHARACTERRAG_PROMPT_TEMPLATE = """## Role:
You are {character_name}. Answer the user's questions as if you are {character_name}.

## Attributes:
### {character_name}'s Belief and Value:
{beliefs}

### {character_name}'s Demographic Information:
{demographics}

### {character_name}'s Psychological Traits:
{traits}

### {character_name}'s Skill and Expertise:
{skills}

### {character_name}'s Social Relationships:
{relationships}

## Knowledge:
{knowledge}
"""


@dataclass
class Persona:
    """
    Persona profile defining the Examinee's psychological ground truth.
    
    Supports both direct system prompts and structured attributes.
    """
    name: str
    description: str = ""
    
    # Direct system prompt (for AnnaAgent)
    system_prompt: Optional[str] = None
    
    # Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    background: Optional[str] = None
    
    # CharacterRAG structured attributes (AMADEUS-style)
    beliefs: Optional[str] = None
    demographics: Optional[str] = None
    psychological_traits: Optional[str] = None
    skills_expertise: Optional[str] = None
    social_relationships: Optional[str] = None
    knowledge: Optional[str] = None
    
    # Personality (ground truth)
    big_five: Optional[Dict[str, int]] = None  # O, C, E, A, N scores (1-5)
    mbti: Optional[str] = None  # e.g., "INTJ"
    
    # Mental health (ground truth)
    depression_level: Optional[int] = None  # 0-4
    suicide_risk: Optional[int] = None  # 0-3
    
    # Additional traits
    traits: Dict[str, Any] = field(default_factory=dict)
    
    # Source metadata
    source_type: str = "custom"  # "annaagent", "characterrag", "custom"
    source_id: Optional[str] = None
    
    def to_system_prompt(self) -> str:
        """Generate a system prompt for the LLM to adopt this persona."""
        # If direct system prompt is provided, use it
        if self.system_prompt:
            return self.system_prompt
        
        # If CharacterRAG attributes are provided, use structured format
        if (
            self.beliefs
            or self.demographics
            or self.psychological_traits
            or self.skills_expertise
            or self.social_relationships
            or self.knowledge
        ):
            return CHARACTERRAG_PROMPT_TEMPLATE.format(
                character_name=self.name,
                beliefs=self.beliefs or "- Not specified",
                demographics=self.demographics or "- Not specified",
                traits=self.psychological_traits or "- Not specified",
                skills=self.skills_expertise or "- Not specified",
                relationships=self.social_relationships or "- Not specified",
                knowledge=self.knowledge or "- Not specified",
            )
        
        # Fallback to basic format
        lines = [
            f"You are {self.name}.",
            f"Description: {self.description}",
        ]
        
        if self.age:
            lines.append(f"Age: {self.age}")
        if self.gender:
            lines.append(f"Gender: {self.gender}")
        if self.occupation:
            lines.append(f"Occupation: {self.occupation}")
        if self.background:
            lines.append(f"Background: {self.background}")
        
        lines.append("")
        lines.append("Stay in character throughout the conversation.")
        lines.append("Respond naturally as this person would, based on their personality and background.")
        
        return "\n".join(lines)


class CharacterLoader:
    """
    Loads character data from AnnaAgent or CharacterRAG sources.
    """
    
    @staticmethod
    def load_from_annaagent(json_path: Union[str, Path], character_id: Optional[str] = None) -> List[Persona]:
        """
        Load personas from AnnaAgent JSON format (D4_prompts.json).
        
        Args:
            json_path: Path to the JSON file
            character_id: Optional specific character ID to load
            
        Returns:
            List of Persona objects
        """
        path = Path(json_path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        personas = []
        for entry in data:
            entry_id = entry.get('id', '')
            
            # Filter by ID if specified
            if character_id and entry_id != character_id:
                continue
            
            # Extract system prompt (actual key in AnnaAgent data is 'prompt')
            system_prompt = entry.get('prompt', '')
            
            # Parse basic info from prompt
            name = CharacterLoader._extract_field(system_prompt, r'性别:\s*(\S+)')
            age_str = CharacterLoader._extract_field(system_prompt, r'年龄:\s*(\d+)')
            occupation = CharacterLoader._extract_field(system_prompt, r'职业:\s*(.+?)(?:\n|$)')
            
            # Parse labels from data
            label = entry.get('label', {})
            depression_level = label.get('drisk')
            suicide_risk = label.get('srisk')

            personas.append(Persona(
                name=f"Patient_{entry_id[:8]}",
                description=entry.get('source', 'AnnaAgent character'),
                system_prompt=system_prompt,
                age=int(age_str) if age_str else None,
                gender=name,
                occupation=occupation,
                depression_level=depression_level,
                suicide_risk=suicide_risk,
                source_type="annaagent",
                source_id=entry_id,
                traits={
                    "record": entry.get("record"),
                    "portrait": entry.get("portrait"),
                }
            ))
        
        return personas
    
    @staticmethod
    def load_from_characterrag(
        profile_path: Union[str, Path],
        character_name: Optional[str] = None,
    ) -> Persona:
        """
        Load persona from CharacterRAG profile TXT file.
        
        Extracts and maps profile sections to the CharacterRAG prompt template:
        - Profile section -> Beliefs and Values
        - Personality/characteristics section -> Psychological Traits
        - Things said/speaking section -> Speech Style
        
        Args:
            profile_path: Path to the profile TXT file (e.g., anya_forger_en.txt)
            character_name: Override character name (uses filename if None)
            
        Returns:
            Persona object with structured attributes
        """
        import warnings
        warnings.warn(
            "load_from_characterrag is deprecated. Use load_from_characterrag_prompt instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = Path(profile_path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract character name from filename if not provided
        if not character_name:
            # e.g., "anya_forger_en.txt" -> "Anya Forger"
            stem = path.stem.replace('_en', '').replace('_', ' ')
            character_name = stem.title()
        
        # Parse structured sections and map to template format
        
        # Beliefs/Values: Extract from Profile and likes/background
        beliefs = CharacterLoader._extract_beliefs(content, character_name)
        
        # Demographic Information: Extract from profile header
        demographics = CharacterLoader._extract_demographics(content)
        
        # Psychological Traits: Extract from Personality section
        traits = CharacterLoader._extract_traits(content)
        
        # Skill and Expertise: Extract from abilities/skills sections
        skills = CharacterLoader._extract_skills(content)
        
        # Social Relationships: Extract from family/relationship mentions
        relationships = CharacterLoader._extract_relationships(content)
        
        # Knowledge: Static chunk (no per-turn retrieval)
        knowledge = CharacterLoader._extract_knowledge(content)
        
        # Extract additional profile info
        age_match = re.search(r'Age:\s*(\d+)', content)
        
        return Persona(
            name=character_name,
            description=f"CharacterRAG profile from {path.name}",
            beliefs=beliefs,
            demographics=demographics,
            psychological_traits=traits,
            skills_expertise=skills,
            social_relationships=relationships,
            knowledge=knowledge,
            age=int(age_match.group(1)) if age_match else None,
            source_type="characterrag",
            source_id=path.stem,
        )

    @staticmethod
    def load_from_characterrag_prompt(
        prompt_path: Union[str, Path],
        character_name: Optional[str] = None,
    ) -> Persona:
        """
        Load persona from a prebuilt CharacterRAG prompt file.
        
        Args:
            prompt_path: Path to the prompt TXT file
            character_name: Override character name
            
        Returns:
            Persona object with system prompt
        """
        path = Path(prompt_path)
        with open(path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        if not character_name:
            match = re.search(r'You are\s+([^.\n]+)', prompt)
            if match:
                character_name = match.group(1).strip()
            else:
                character_name = path.stem.replace('_en', '').replace('_', ' ').title()
        
        return Persona(
            name=character_name,
            description=f"CharacterRAG prompt from {path.name}",
            system_prompt=prompt,
            source_type="characterrag_prompt",
            source_id=path.stem,
        )
    
    @staticmethod
    def _extract_beliefs(content: str, character_name: str) -> str:
        """Extract beliefs/values from profile sections."""
        beliefs = []
        
        # Look for family, likes, values, or explicit belief sections
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Extract likes
            if stripped.startswith('- Likes:'):
                likes = stripped.replace('- Likes:', '').strip()
                top_likes = ', '.join(likes.split(',')[:5])  # First 5 likes
                beliefs.append(f"- Values and enjoys: {top_likes}")
            
            # Extract family info (suggests family values)
            elif 'Family' in stripped and stripped.startswith('-'):
                beliefs.append(f"- {stripped[1:].strip()}")
            
            # Look for specific value-related keywords
            elif any(kw in stripped.lower() for kw in ['treasures', 'precious', 'important', 'believes', 'values']):
                if stripped.startswith('-'):
                    beliefs.append(stripped)
        
        if not beliefs:
            beliefs.append(f"- Values family relationships")
            beliefs.append(f"- Curious and enjoys learning")
        
        return '\n'.join(beliefs[:5])  # Limit to 5 beliefs

    @staticmethod
    def _extract_demographics(content: str) -> str:
        """Extract demographic information from the profile header."""
        demographics = []
        lines = content.split('\n')
        in_profile = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') and 'Profile' in stripped:
                in_profile = True
                continue
            if in_profile and stripped.startswith('#') and 'Profile' not in stripped:
                break
            if in_profile and stripped.startswith('-'):
                if any(key in stripped for key in ['Age:', 'Nationality:', 'Physical:', 'School:', 'Occupation:', 'First-person']):
                    demographics.append(stripped)
        
        return '\n'.join(demographics[:6]) if demographics else '- Not specified'
    
    @staticmethod
    def _extract_traits(content: str) -> str:
        """Extract psychological traits from Personality section."""
        traits = []
        
        # Find Personality section
        in_personality = False
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            # Check for personality section header
            if 'Personality' in stripped and stripped.startswith('#'):
                in_personality = True
                continue
            
            # Check for next major section
            if in_personality and stripped.startswith('#') and 'Personality' not in stripped:
                break
            
            # Collect bullet points in personality section
            if in_personality and stripped.startswith('-'):
                # Clean up the trait description
                trait = stripped
                # Limit line length for readability
                if len(trait) > 150:
                    trait = trait[:147] + '...'
                traits.append(trait)
        
        # Also look for telepathy/special abilities
        for line in content.split('\n'):
            if 'telepathy' in line.lower() or 'psychic' in line.lower():
                if line.strip().startswith('-'):
                    traits.append(line.strip()[:150])
                    break
        
        return '\n'.join(traits[:6]) if traits else '- Not specified'

    @staticmethod
    def _extract_skills(content: str) -> str:
        """Extract skills and expertise from abilities or skills sections."""
        skills = []
        lines = content.split('\n')
        in_skills = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') and any(key in stripped for key in ['Abilities', 'Skills', 'Grades', 'Special Moves']):
                in_skills = True
                continue
            if in_skills and stripped.startswith('#') and not any(key in stripped for key in ['Abilities', 'Skills', 'Grades', 'Special Moves']):
                break
            if in_skills and stripped.startswith('-'):
                if len(stripped) > 150:
                    stripped = stripped[:147] + '...'
                skills.append(stripped)
        
        return '\n'.join(skills[:6]) if skills else '- Not specified'

    @staticmethod
    def _extract_relationships(content: str) -> str:
        """Extract social relationships from profile lines."""
        relationships = []
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('-') and any(key in stripped for key in ['Family', 'friends', 'relationship', 'adoptive']):
                relationships.append(stripped)
        
        return '\n'.join(relationships[:5]) if relationships else '- Not specified'

    @staticmethod
    def _extract_knowledge(content: str, max_sections: int = 2) -> str:
        """Extract a static knowledge chunk from the first few top-level sections."""
        sections = []
        lines = content.split('\n')
        current = []
        in_section = False
        
        for line in lines:
            if line.startswith('# '):
                if current:
                    sections.append('\n'.join(current).strip())
                    current = []
                in_section = True
            if in_section:
                current.append(line)
        
        if current:
            sections.append('\n'.join(current).strip())
        
        selected = [sec for sec in sections if sec][:max_sections]
        return '\n\n'.join(selected) if selected else '- Not specified'
    
    @staticmethod
    def _extract_speech_style(content: str) -> str:
        """Deprecated: retained for compatibility in older callers."""
        return '- Not specified'
    
    @staticmethod
    def load_characterrag_segments(
        profile_dir: Union[str, Path],
        character_name: str,
        segment_ids: List[str],
    ) -> Persona:
        """
        Load persona from CharacterRAG by selecting specific profile segments.
        
        This method allows selecting specific parts of the character profile
        for fine-grained persona construction.
        
        Args:
            profile_dir: Directory containing character profile files
            character_name: Name of the character
            segment_ids: List of segment identifiers to include
            
        Returns:
            Persona with selected segments
        """
        import warnings
        warnings.warn(
            "load_characterrag_segments is deprecated. Use prebuilt prompt files instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        profile_dir = Path(profile_dir)
        
        # Find profile file
        profile_files = list(profile_dir.glob(f"*{character_name.lower().replace(' ', '_')}*.txt"))
        if not profile_files:
            raise FileNotFoundError(f"No profile found for {character_name} in {profile_dir}")
        
        profile_path = profile_files[0]
        with open(profile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract segments by header
        selected_content = []
        for seg_id in segment_ids:
            pattern = rf'#+\s*\d*\.?\d*\.?\s*{re.escape(seg_id)}[:\s]*(.*?)(?=\n#+|\Z)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                selected_content.append(match.group(0).strip())
        
        combined = "\n\n".join(selected_content)
        
        return Persona(
            name=character_name.title(),
            description=f"Selected segments from {profile_path.name}",
            system_prompt=f"You are {character_name.title()}.\n\n{combined}\n\nStay in character.",
            source_type="characterrag",
            source_id=f"{profile_path.stem}:{','.join(segment_ids)}",
        )
    
    @staticmethod
    def _extract_field(text: str, pattern: str) -> Optional[str]:
        """Extract a field using regex."""
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None


@dataclass
class TATResponse:
    """Response to a TAT image."""
    image_id: str
    narrative: str
    
    
@dataclass
class RorschachResponse:
    """Response to a Rorschach card."""
    card_number: int
    perception: str  # What they see
    inquiry: Optional[str] = None  # Why they see it (location, determinants)


@dataclass 
class SCTResponse:
    """Response to an SCT sentence stem."""
    stem_id: str
    stem: str
    completion: str


class Examinee:
    """
    Examinee class representing the LLM under assessment.
    
    X = M | P, where M is the LLM and P is the Persona.
    
    Generates responses to projective test stimuli:
    - TAT narratives
    - Rorschach perceptions
    - SCT completions
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        persona: Persona,
        generation_config: Optional[GenerationConfig] = None,
        extra_system_messages: Optional[List[str]] = None,
        prefix_history: Optional[List["Message"]] = None,
    ):
        """
        Initialize the Examinee.

        Args:
            llm: The LLM model to use
            persona: The persona profile
            generation_config: Optional generation settings
            extra_system_messages: Additional system messages inserted AFTER the
                persona system prompt (e.g. social-desirability framing). Each
                entry becomes its own `{"role": "system", ...}` message in turn
                order, so downstream chat templates can render them as an extra
                situational system turn without polluting the persona definition.
            prefix_history: Multi-turn messages inserted AFTER the system block
                and BEFORE every elicitation turn. Used to "seed" the subject
                with a prior conversation (AnnaAgent counselor log, CharacterRAG
                Q&A) so subsequent projective/questionnaire responses are
                conditioned on that context.
        """
        self.llm = llm
        self.persona = persona
        self.config = generation_config or GenerationConfig(
            max_tokens=1024,
            temperature=0.8,
        )
        self._system_prompt = persona.to_system_prompt()
        self._extra_system_messages = list(extra_system_messages or [])
        self._prefix_history = list(prefix_history or [])

    def _system_messages(self) -> List["Message"]:
        """Return [persona_sys, *extra_sys] as Message objects."""
        msgs = [Message(role="system", content=self._system_prompt)]
        for extra in self._extra_system_messages:
            msgs.append(Message(role="system", content=extra))
        return msgs

    def _compose_messages(self, *turns: "Message") -> List["Message"]:
        """
        Compose a full message list:
            [persona_sys, *extra_sys, *prefix_history, *turns]
        """
        return self._system_messages() + list(self._prefix_history) + list(turns)
    
    @classmethod
    def from_annaagent(
        cls,
        llm: BaseLLM,
        json_path: Union[str, Path],
        character_id: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> "Examinee":
        """
        Create Examinee from AnnaAgent character data.
        
        Args:
            llm: The LLM model to use
            json_path: Path to D4_prompts.json
            character_id: ID of the character to load
            generation_config: Optional generation settings
        """
        personas = CharacterLoader.load_from_annaagent(json_path, character_id)
        if not personas:
            raise ValueError(f"Character {character_id} not found in {json_path}")
        return cls(llm, personas[0], generation_config)
    
    @classmethod
    def from_characterrag(
        cls,
        llm: BaseLLM,
        profile_path: Union[str, Path],
        character_name: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> "Examinee":
        """
        Create Examinee from CharacterRAG profile.
        
        Args:
            llm: The LLM model to use
            profile_path: Path to profile TXT file
            character_name: Override character name
            generation_config: Optional generation settings
        """
        profile_path = Path(profile_path)
        if profile_path.suffix == ".txt" and profile_path.name.endswith(".prompt.txt"):
            prompt_path = profile_path
        else:
            prompt_path = profile_path.with_suffix(".prompt.txt")
        
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"CharacterRAG prompt not found: {prompt_path}. "
                "Generate prompts first or pass a .prompt.txt path."
            )
        
        persona = CharacterLoader.load_from_characterrag_prompt(prompt_path, character_name)
        return cls(llm, persona, generation_config)
    
    def respond_tat(
        self,
        image: TATImage,
        custom_prompt: Optional[str] = None,
    ) -> TATResponse:
        """
        Generate a narrative response to a TAT image.
        
        Args:
            image: The TAT image stimulus
            custom_prompt: Optional custom prompt (uses default if None)
            
        Returns:
            TATResponse with the generated narrative
        """
        prompt = custom_prompt or TAT_PROMPT_TEMPLATE
        
        messages = self._compose_messages(
            Message(role="user", content=prompt),
        )
        
        narrative = self.llm.generate_with_image(
            messages=messages,
            image_path=image.path,
            config=self.config,
        )
        
        return TATResponse(
            image_id=image.id,
            narrative=narrative,
        )
    
    def respond_tat_batch(
        self,
        images: List[TATImage],
        custom_prompt: Optional[str] = None,
    ) -> List[TATResponse]:
        """Generate responses to multiple TAT images."""
        return [self.respond_tat(img, custom_prompt) for img in images]
    
    def respond_rorschach(
        self,
        cards: List[RorschachCard],
        include_inquiry: bool = True,
    ) -> List[RorschachResponse]:
        """
        Generate responses to Rorschach cards in a multi-turn dialogue.
        
        The Examinee sees cards sequentially, with context from previous responses.
        
        Args:
            cards: List of Rorschach cards (typically 10, in order)
            include_inquiry: Whether to include inquiry phase for each response
            
        Returns:
            List of RorschachResponse objects
        """
        responses = []
        conversation_history = self._system_messages() + list(self._prefix_history)
        
        for idx, card in enumerate(cards):
            # Determine prompt based on position
            if idx == 0:
                prompt = RORSCHACH_FREE_ASSOCIATION_PROMPT
            else:
                prompt = RORSCHACH_FREE_ASSOCIATION_PROMPT
            
            # Add user message with card image
            messages = conversation_history + [
                Message(role="user", content=prompt),
            ]
            
            # Generate perception
            perception = self.llm.generate_with_image(
                messages=messages,
                image_path=card.path,
                config=self.config,
            )
            
            # Update history
            conversation_history.append(Message(role="user", content=prompt))
            conversation_history.append(Message(role="assistant", content=perception))
            
            # Optional inquiry phase
            inquiry = None
            if include_inquiry:
                inquiry_messages = conversation_history + [
                    Message(role="user", content="Can you tell me more about what you saw? What made it look that way?"),
                ]
                inquiry = self.llm.generate(inquiry_messages, self.config)
                conversation_history.append(Message(role="user", content="Can you tell me more?"))
                conversation_history.append(Message(role="assistant", content=inquiry))
            
            responses.append(RorschachResponse(
                card_number=card.card_number,
                perception=perception,
                inquiry=inquiry,
            ))
        
        return responses
    
    def respond_sct(
        self,
        stem: SCTStem,
        custom_prompt: Optional[str] = None,
    ) -> SCTResponse:
        """
        Complete a sentence stem.
        
        Args:
            stem: The sentence stem to complete
            custom_prompt: Optional custom prompt template
            
        Returns:
            SCTResponse with the completion
        """
        prompt_template = custom_prompt or SCT_PROMPT_TEMPLATE
        prompt = prompt_template.format(stem=stem.stem)

        messages = self._compose_messages(
            Message(role="user", content=prompt),
        )
        completion = self.llm.generate(messages=messages, config=self.config)
        
        return SCTResponse(
            stem_id=stem.id,
            stem=stem.stem,
            completion=completion,
        )
    
    def respond_sct_batch(
        self,
        stems: List[SCTStem],
        custom_prompt: Optional[str] = None,
    ) -> List[SCTResponse]:
        """Complete multiple sentence stems."""
        return [self.respond_sct(stem, custom_prompt) for stem in stems]
    
    def run_full_assessment(
        self,
        tat_images: List[TATImage],
        rorschach_cards: List[RorschachCard],
        sct_stems: List[SCTStem],
    ) -> Dict[str, Any]:
        """
        Run the complete projective test battery.
        
        Returns:
            Dict with all responses organized by test type
        """
        return {
            "persona": self.persona.name,
            "source_type": self.persona.source_type,
            "source_id": self.persona.source_id,
            "tat_responses": self.respond_tat_batch(tat_images),
            "rorschach_responses": self.respond_rorschach(rorschach_cards),
            "sct_responses": self.respond_sct_batch(sct_stems),
        }
