"""
Base LLM Interface

Abstract base class for LLM clients used in GenPT.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: Union[str, List[Dict[str, Any]]]  # Text or multimodal content
    
    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: Optional[int] = None  # None = no limit, model generates until EOS
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_sequences: Optional[List[str]] = None
    

class BaseLLM(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name or path of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text response from messages.
        
        Args:
            messages: List of conversation messages
            config: Generation configuration
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_with_image(
        self,
        messages: List[Message],
        image_path: Union[str, Path],
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text response from messages with an image.
        
        Args:
            messages: List of conversation messages
            image_path: Path to the image file
            config: Generation configuration
            
        Returns:
            Generated text response
        """
        pass
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Simple chat interface for single-turn conversations.
        
        Args:
            user_message: User's input message
            system_prompt: Optional system prompt
            config: Generation configuration
            
        Returns:
            Generated response
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))
        return self.generate(messages, config)
    
    def chat_with_image(
        self,
        user_message: str,
        image_path: Union[str, Path],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Simple chat interface for single-turn conversations with an image.
        
        Args:
            user_message: User's input message
            image_path: Path to the image file
            system_prompt: Optional system prompt
            config: Generation configuration
            
        Returns:
            Generated response
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))
        return self.generate_with_image(messages, image_path, config)
