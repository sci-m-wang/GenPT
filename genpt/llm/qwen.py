"""
Qwen LLM Clients

Provides both VL (Vision-Language) and text-only clients.
Supports vLLM API (OpenAI-compatible) and local inference.
"""

import re
import logging
from typing import List, Optional, Union
from pathlib import Path
import base64

from .base import BaseLLM, Message, GenerationConfig

logger = logging.getLogger("genpt.llm")


def strip_thinking(text: str) -> tuple[str, str]:
    """
    Strip <think>...</think> blocks from Qwen3 thinking mode output.

    Returns:
        (answer_text, thinking_text) — thinking_text is empty when there
        was no <think> block.
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end():].strip()
        return answer, thinking
    return text.strip(), ""


class QwenVLClient(BaseLLM):
    """
    Client for Qwen VL (Vision-Language) models.

    Primary backend: vLLM OpenAI-compatible API.
    Fallback: local transformers inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        use_api: bool = True,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.use_api = use_api
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"
        self.enable_thinking = enable_thinking

        self._model = None
        self._processor = None
        self._api_client = None

    # ── API helpers ───────────────────────────────────────────

    def _init_api_client(self):
        if self._api_client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._api_client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _get_image_mime_type(image_path: Union[str, Path]) -> str:
        ext = Path(image_path).suffix.lower()
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp"}.get(ext, "image/jpeg")

    # ── Local model loading ───────────────────────────────────

    def _load_local_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoConfig, AutoProcessor
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch")

        logger.info("Loading VL model %s ...", self.model_name)
        cfg = AutoConfig.from_pretrained(self.model_name)
        model_type = getattr(cfg, "model_type", "")
        use_qwen3 = "qwen3_vl" in model_type or "Qwen3" in self.model_name

        if use_qwen3:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = ModelCls.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self.device,
        )
        logger.info("VL model loaded.")

    # ── Generate implementations ──────────────────────────────

    def generate(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        if self.use_api:
            return self._generate_api(messages, config)
        return self._generate_local(messages, config)

    def generate_with_image(self, messages: List[Message], image_path: Union[str, Path],
                            config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        if self.use_api:
            return self._generate_with_image_api(messages, image_path, config)
        return self._generate_with_image_local(messages, image_path, config)

    # ── Local generation ──────────────────────────────────────

    def _generate_local(self, messages: List[Message], config: GenerationConfig) -> str:
        self._load_local_model()
        formatted = [msg.to_dict() for msg in messages]
        text = self._processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], return_tensors="pt").to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        gen_kwargs = dict(
            temperature=max(config.temperature, 1e-7), top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.temperature > 0,
        )
        if config.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = config.max_tokens
        else:
            model_max = getattr(self._model.config, "max_position_embeddings", 40960)
            gen_kwargs["max_new_tokens"] = max(model_max - input_len, 1024)
        outputs = self._model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][input_len:]
        response = self._processor.decode(generated, skip_special_tokens=True)
        answer, _ = strip_thinking(response)
        return answer

    def _generate_with_image_local(self, messages: List[Message], image_path: Union[str, Path],
                                   config: GenerationConfig) -> str:
        self._load_local_model()
        from qwen_vl_utils import process_vision_info

        formatted = []
        for msg in messages:
            if msg.role == "user":
                content = [{"type": "image", "image": str(image_path)},
                           {"type": "text", "text": msg.content if isinstance(msg.content, str) else str(msg.content)}]
                formatted.append({"role": "user", "content": content})
            else:
                formatted.append(msg.to_dict())

        text = self._processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(formatted)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs,
                                 return_tensors="pt").to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        gen_kwargs = dict(
            temperature=max(config.temperature, 1e-7), top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.temperature > 0,
        )
        if config.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = config.max_tokens
        else:
            model_max = getattr(self._model.config, "max_position_embeddings", 40960)
            gen_kwargs["max_new_tokens"] = max(model_max - input_len, 1024)
        outputs = self._model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][input_len:]
        response = self._processor.decode(generated, skip_special_tokens=True)
        answer, _ = strip_thinking(response)
        return answer

    # ── API generation ────────────────────────────────────────

    def _generate_api(self, messages: List[Message], config: GenerationConfig) -> str:
        self._init_api_client()
        formatted = [msg.to_dict() for msg in messages]
        extra_body = {}
        if self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        api_kwargs = dict(
            model=self.model_name, messages=formatted,
            temperature=config.temperature, top_p=config.top_p,
            extra_body=extra_body if extra_body else None,
        )
        if config.max_tokens is not None:
            api_kwargs["max_tokens"] = config.max_tokens
        resp = self._api_client.chat.completions.create(**api_kwargs)
        raw = resp.choices[0].message.content
        answer, _ = strip_thinking(raw)
        return answer

    def _generate_with_image_api(self, messages: List[Message], image_path: Union[str, Path],
                                 config: GenerationConfig) -> str:
        self._init_api_client()
        b64 = self._encode_image(image_path)
        mime = self._get_image_mime_type(image_path)
        # Find index of the last user message; only inject the image there.
        # Prior user turns in multi-turn (e.g. Rorschach) already had their
        # images consumed and are now text-only history.
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_idx = i
                break
        formatted = []
        for i, msg in enumerate(messages):
            if msg.role == "user" and i == last_user_idx:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": msg.content if isinstance(msg.content, str) else str(msg.content)},
                ]
                formatted.append({"role": "user", "content": content})
            else:
                formatted.append(msg.to_dict())
        extra_body = {}
        if self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        api_kwargs = dict(
            model=self.model_name, messages=formatted,
            temperature=config.temperature, top_p=config.top_p,
            extra_body=extra_body if extra_body else None,
        )
        if config.max_tokens is not None:
            api_kwargs["max_tokens"] = config.max_tokens
        resp = self._api_client.chat.completions.create(**api_kwargs)
        raw = resp.choices[0].message.content
        answer, _ = strip_thinking(raw)
        return answer


class QwenTextClient(BaseLLM):
    """
    Client for Qwen text-only models (e.g. Qwen3-8B).

    Primary backend: vLLM OpenAI-compatible API.
    Fallback: local transformers inference.
    Supports PEFT LoRA adapter loading and hot-swapping.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        use_api: bool = True,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.use_api = use_api
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"
        self.enable_thinking = enable_thinking

        self._model = None
        self._tokenizer = None
        self._api_client = None
        self._loaded_adapters: dict[str, str] = {}  # name -> path
        self._active_adapter: Optional[str] = None

    # ── API helpers ───────────────────────────────────────────

    def _init_api_client(self):
        if self._api_client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._api_client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    # ── Local model loading ───────────────────────────────────

    def _load_local_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch")

        logger.info("Loading text model %s ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self.device,
        )
        logger.info("Text model loaded.")

    # ── LoRA adapter management ───────────────────────────────

    def load_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load a PEFT LoRA adapter onto the model.

        Args:
            adapter_path: Path to a directory containing adapter_config.json.
            adapter_name: Unique name to reference this adapter.
        """
        self._load_local_model()
        from peft import PeftModel

        if adapter_name in self._loaded_adapters:
            logger.info("Adapter '%s' already loaded, skipping", adapter_name)
            return

        if not self._loaded_adapters:
            # First adapter — wrap model with PeftModel
            logger.info("Loading initial LoRA adapter '%s' from %s", adapter_name, adapter_path)
            self._model = PeftModel.from_pretrained(
                self._model, adapter_path, adapter_name=adapter_name
            )
        else:
            # Additional adapter
            logger.info("Loading additional LoRA adapter '%s' from %s", adapter_name, adapter_path)
            self._model.load_adapter(adapter_path, adapter_name=adapter_name)

        self._loaded_adapters[adapter_name] = adapter_path
        self._active_adapter = adapter_name
        logger.info("Active adapter: %s", adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        """Switch the active LoRA adapter.

        Args:
            adapter_name: Name of a previously loaded adapter.
        """
        if adapter_name not in self._loaded_adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not loaded. "
                f"Available: {list(self._loaded_adapters)}"
            )
        if self._active_adapter != adapter_name:
            self._model.set_adapter(adapter_name)
            self._active_adapter = adapter_name
            logger.info("Switched adapter to: %s", adapter_name)

    def disable_adapter(self) -> None:
        """Disable all adapters, falling back to the base model."""
        if self._loaded_adapters and self._active_adapter is not None:
            self._model.disable_adapter_layers()
            self._active_adapter = None
            logger.info("All adapters disabled (base model)")

    def enable_adapter(self) -> None:
        """Re-enable adapters after disable_adapter()."""
        if self._loaded_adapters:
            self._model.enable_adapter_layers()
            logger.info("Adapters re-enabled")

    # ── Generate implementation ───────────────────────────────

    def generate(self, messages: List[Message], config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        if self.use_api:
            return self._generate_api(messages, config)
        return self._generate_local(messages, config)

    def generate_with_image(self, messages: List[Message], image_path: Union[str, Path],
                            config: Optional[GenerationConfig] = None) -> str:
        """Text-only model — ignore image, fall back to text generation."""
        logger.warning("QwenTextClient does not support images; ignoring image_path.")
        return self.generate(messages, config)

    # ── Local generation ──────────────────────────────────────

    def _generate_local(self, messages: List[Message], config: GenerationConfig) -> str:
        self._load_local_model()
        formatted = [msg.to_dict() for msg in messages]
        text = self._tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            temperature=max(config.temperature, 1e-7), top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.temperature > 0,
        )
        if config.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = config.max_tokens
        else:
            # HF default max_length=20 is useless; use model context window
            model_max = getattr(self._model.config, "max_position_embeddings", 40960)
            gen_kwargs["max_new_tokens"] = max(model_max - input_len, 1024)

        outputs = self._model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][input_len:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)
        answer, _ = strip_thinking(response)
        return answer

    # ── API generation ────────────────────────────────────────

    def _generate_api(self, messages: List[Message], config: GenerationConfig) -> str:
        self._init_api_client()
        formatted = [msg.to_dict() for msg in messages]
        extra_body = {}
        if self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        api_kwargs = dict(
            model=self.model_name, messages=formatted,
            temperature=config.temperature, top_p=config.top_p,
            extra_body=extra_body if extra_body else None,
        )
        if config.max_tokens is not None:
            api_kwargs["max_tokens"] = config.max_tokens
        resp = self._api_client.chat.completions.create(**api_kwargs)
        raw = resp.choices[0].message.content
        answer, _ = strip_thinking(raw)
        return answer


def create_client_from_config(model_config) -> BaseLLM:
    """
    Factory function: create the appropriate LLM client from a ModelConfig.

    Uses model_path for local loading, or api_base for API mode.

    Args:
        model_config: A config.ModelConfig instance.

    Returns:
        QwenVLClient for multimodal models, QwenTextClient for text-only.
    """
    use_api = bool(model_config.api_base)
    model_name = model_config.model_path or model_config.model_name

    if model_config.is_multimodal:
        return QwenVLClient(
            model_name=model_name,
            use_api=use_api,
            api_base=model_config.api_base,
            api_key=model_config.api_key,
            enable_thinking=model_config.enable_thinking,
        )
    else:
        return QwenTextClient(
            model_name=model_name,
            use_api=use_api,
            api_base=model_config.api_base,
            api_key=model_config.api_key,
            enable_thinking=model_config.enable_thinking,
        )
