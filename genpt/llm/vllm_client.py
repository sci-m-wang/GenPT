"""
vLLM Offline Batch Clients for GenPT
====================================

Provides high-throughput batch inference with vLLM for both the text model
(Qwen3-8B) and the vision-language model (Qwen3-VL-8B-Instruct).

Design goals
------------
* ``VLLMTextClient`` — batched text generation for the Interpreter and
  Diagnostician stages.  Supports Qwen3 thinking mode and LoRA adapter
  hot-swapping at generation time (``LoRARequest``).
* ``VLLMVisionClient`` — batched multimodal generation for the Examinee stage
  (TAT / Rorschach).
* Both clients are drop-in replacements for :class:`genpt.llm.base.BaseLLM`.
* The real throughput win comes from ``generate_batch``/``chat_batch``:
  many prompts are scheduled together by vLLM's continuous batching.

The clients lazily instantiate the underlying ``vllm.LLM`` the first time a
request is made, so importing the module is cheap (no CUDA init at import).
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .base import BaseLLM, GenerationConfig, Message

logger = logging.getLogger("genpt.llm.vllm")


def strip_thinking(text: str) -> tuple[str, str]:
    """Strip ``<think>...</think>`` blocks from Qwen3 thinking-mode outputs.

    Returns
    -------
    (answer, thinking)
        ``thinking`` is the content of the think block (if any), ``answer`` is
        everything that follows it.
    """
    if text is None:
        return "", ""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return text[m.end():].strip(), m.group(1).strip()
    return text.strip(), ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _messages_to_dicts(messages: Sequence[Message | Dict]) -> List[Dict]:
    out: List[Dict] = []
    for msg in messages:
        if isinstance(msg, Message):
            out.append(msg.to_dict())
        else:
            out.append(dict(msg))
    return out


def _build_sampling_params(config: Optional[GenerationConfig],
                           default_max_tokens: int = 2048):
    from vllm import SamplingParams

    cfg = config or GenerationConfig()
    max_tokens = cfg.max_tokens if cfg.max_tokens is not None else default_max_tokens
    return SamplingParams(
        temperature=max(cfg.temperature, 0.0),
        top_p=cfg.top_p,
        max_tokens=max_tokens,
        repetition_penalty=cfg.repetition_penalty,
        stop=cfg.stop_sequences,
    )


# ---------------------------------------------------------------------------
# Text client
# ---------------------------------------------------------------------------


@dataclass
class _LoRASpec:
    name: str
    path: str
    int_id: int


class VLLMTextClient(BaseLLM):
    """High-throughput text client backed by vLLM offline batch inference.

    Parameters
    ----------
    model_name : str
        Path or HF id of the model.
    tensor_parallel_size : int
        vLLM TP size.  On 8×A100-40GB we use 1 so that each sampler worker can
        own a single GPU for parallel collection jobs.
    gpu_memory_utilization : float
        Fraction of GPU memory vLLM can use.
    max_model_len : int
        Maximum context window.  Qwen3-8B supports 40960 tokens.
    enable_thinking : bool
        Whether to ask the Qwen3 chat template for thinking mode.
    enable_lora : bool
        Enable LoRA support in the underlying vLLM engine (required if you
        want to hot-swap task-specific adapters at inference time).
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        enable_thinking: bool = True,
        enable_lora: bool = False,
        max_loras: int = 4,
        max_lora_rank: int = 32,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        max_num_seqs: Optional[int] = None,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.chunk_size = chunk_size  # cap per engine.generate() call

        self._llm = None
        self._tokenizer = None
        self._lora_specs: Dict[str, _LoRASpec] = {}
        self._active_lora: Optional[str] = None
        self._next_lora_id: int = 1

    # ----- lazy engine init -----

    def _engine(self):
        if self._llm is not None:
            return self._llm
        from vllm import LLM

        logger.info(
            "Initialising vLLM text engine: model=%s tp=%d gpu_mem=%.2f max_len=%d lora=%s",
            self.model_name, self.tensor_parallel_size,
            self.gpu_memory_utilization, self.max_model_len, self.enable_lora,
        )
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            trust_remote_code=True,
            enable_lora=self.enable_lora,
            max_loras=self.max_loras if self.enable_lora else 1,
            max_lora_rank=self.max_lora_rank if self.enable_lora else 16,
            enforce_eager=self.enforce_eager,
            **({"max_num_seqs": self.max_num_seqs}
               if self.max_num_seqs is not None else {}),
        )
        self._tokenizer = self._llm.get_tokenizer()
        return self._llm

    # ----- chat template -----

    def _render_prompt(self, messages: Sequence[Message | Dict]) -> str:
        self._engine()  # ensure tokenizer
        msg_dicts = _messages_to_dicts(messages)
        try:
            return self._tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Older tokenizers do not accept ``enable_thinking``.
            return self._tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True,
            )

    # ----- LoRA management -----

    def load_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Register a LoRA adapter with the engine.

        The adapter is not activated until ``set_adapter()`` is called (or a
        ``lora_request`` is passed explicitly to generate).
        """
        if not self.enable_lora:
            raise RuntimeError(
                "enable_lora=False — re-create the client with enable_lora=True "
                "to hot-swap adapters."
            )
        if adapter_name in self._lora_specs:
            return
        spec = _LoRASpec(adapter_name, str(adapter_path), self._next_lora_id)
        self._next_lora_id += 1
        self._lora_specs[adapter_name] = spec
        logger.info("Registered LoRA adapter '%s' -> %s (id=%d)",
                    adapter_name, adapter_path, spec.int_id)

    def set_adapter(self, adapter_name: Optional[str]) -> None:
        if adapter_name is not None and adapter_name not in self._lora_specs:
            raise KeyError(f"Adapter '{adapter_name}' not loaded.")
        self._active_lora = adapter_name

    def _lora_request(self, explicit: Optional[str] = None):
        name = explicit if explicit is not None else self._active_lora
        if not name:
            return None
        from vllm.lora.request import LoRARequest
        spec = self._lora_specs[name]
        return LoRARequest(lora_name=spec.name, lora_int_id=spec.int_id,
                           lora_path=spec.path)

    # ----- single-message API (BaseLLM) -----

    def generate(self, messages: List[Message],
                 config: Optional[GenerationConfig] = None) -> str:
        return self.generate_batch([messages], config)[0]

    def generate_with_image(self, messages, image_path, config=None):  # noqa: D401
        logger.warning("VLLMTextClient does not support images; ignoring.")
        return self.generate(messages, config)

    # ----- batched API (the real performance path) -----

    def generate_batch(
        self,
        batches: Sequence[Sequence[Message | Dict]],
        config: Optional[GenerationConfig] = None,
        *,
        adapter: Optional[str] = None,
        strip_think: bool = True,
        return_thinking: bool = False,
    ) -> List[str] | List[tuple[str, str]]:
        """Run a batch of conversations through the engine in one call.

        When ``return_thinking`` is True, each element is a ``(answer, thinking)``
        tuple.
        """
        engine = self._engine()
        params = _build_sampling_params(config)
        prompts = [self._render_prompt(msgs) for msgs in batches]
        lora_req = self._lora_request(adapter)

        # Optionally chunk the submission to avoid peak KV / scheduler pressure.
        if self.chunk_size and len(prompts) > self.chunk_size:
            all_outputs = []
            total_chunks = (len(prompts) + self.chunk_size - 1) // self.chunk_size
            for ci in range(total_chunks):
                s = ci * self.chunk_size
                e = min(s + self.chunk_size, len(prompts))
                logger.info("generate_batch chunk %d/%d (%d prompts)",
                            ci + 1, total_chunks, e - s)
                chunk_out = engine.generate(
                    prompts[s:e], params, lora_request=lora_req, use_tqdm=False,
                )
                all_outputs.extend(chunk_out)
            outputs = all_outputs
        else:
            outputs = engine.generate(prompts, params, lora_request=lora_req,
                                      use_tqdm=False)
        # vLLM returns in submission order
        results: List[Any] = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            if strip_think:
                ans, think = strip_thinking(text)
            else:
                ans, think = text, ""
            if return_thinking:
                results.append((ans, think))
            else:
                results.append(ans)
        return results

    def chat_batch(
        self,
        user_messages: Sequence[str],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        *,
        adapter: Optional[str] = None,
        return_thinking: bool = False,
    ) -> List[str] | List[tuple[str, str]]:
        batches = []
        for user in user_messages:
            msgs: List[Dict] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user})
            batches.append(msgs)
        return self.generate_batch(batches, config, adapter=adapter,
                                   return_thinking=return_thinking)


# ---------------------------------------------------------------------------
# Vision client
# ---------------------------------------------------------------------------


class VLLMVisionClient(BaseLLM):
    """Batched multimodal client for Qwen3-VL via vLLM."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        enable_thinking: bool = False,
        dtype: str = "bfloat16",
        limit_mm_per_prompt: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.dtype = dtype
        self.limit_mm_per_prompt = limit_mm_per_prompt

        self._llm = None
        self._processor = None

    def _engine(self):
        if self._llm is not None:
            return self._llm
        from vllm import LLM

        logger.info(
            "Initialising vLLM VL engine: model=%s tp=%d gpu_mem=%.2f max_len=%d",
            self.model_name, self.tensor_parallel_size,
            self.gpu_memory_utilization, self.max_model_len,
        )
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
        )
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Could not load processor: %s", e)
            self._processor = None
        return self._llm

    # ----- chat template -----

    def _render_prompt(self, messages: Sequence[Dict]) -> str:
        self._engine()
        if self._processor is None:
            # Fallback: use engine's tokenizer
            tok = self._llm.get_tokenizer()
            try:
                return tok.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                return tok.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True,
                )
        try:
            return self._processor.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self._processor.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True,
            )

    # ----- image helpers -----

    @staticmethod
    def _load_pil_image(path: Union[str, Path]):
        from PIL import Image
        img = Image.open(str(path))
        img.load()
        return img.convert("RGB")

    def _prepare(
        self,
        messages: Sequence[Message | Dict],
        image_paths: Optional[Sequence[Union[str, Path]]] = None,
    ) -> Dict[str, Any]:
        """Build a vLLM prompt dict (``{"prompt": ..., "multi_modal_data": ...}``).

        ``image_paths`` are inserted into the first user turn that does not
        already contain image placeholders; this keeps call sites simple for
        the common single-user-turn case.
        """
        msg_dicts = _messages_to_dicts(messages)
        images: List[Any] = []

        if image_paths:
            images = [self._load_pil_image(p) for p in image_paths]
            # Inject image placeholders into the first user message
            for m in msg_dicts:
                if m["role"] != "user":
                    continue
                content = m["content"]
                if isinstance(content, str):
                    m["content"] = [
                        *({"type": "image"} for _ in images),
                        {"type": "text", "text": content},
                    ]
                elif isinstance(content, list):
                    has_image = any(c.get("type") == "image" for c in content
                                     if isinstance(c, dict))
                    if not has_image:
                        m["content"] = [
                            *({"type": "image"} for _ in images),
                            *content,
                        ]
                break

        prompt = self._render_prompt(msg_dicts)
        req: Dict[str, Any] = {"prompt": prompt}
        if images:
            req["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}
        return req

    # ----- BaseLLM API -----

    def generate(self, messages: List[Message],
                 config: Optional[GenerationConfig] = None) -> str:
        return self.generate_batch([messages], [None], config)[0]

    def generate_with_image(self, messages, image_path, config=None):
        return self.generate_batch([messages], [[image_path]], config)[0]

    # ----- batch API -----

    def generate_batch(
        self,
        batches: Sequence[Sequence[Message | Dict]],
        image_batches: Optional[Sequence[Optional[Sequence[Union[str, Path]]]]] = None,
        config: Optional[GenerationConfig] = None,
        *,
        strip_think: bool = True,
    ) -> List[str]:
        engine = self._engine()
        params = _build_sampling_params(config, default_max_tokens=1024)

        if image_batches is None:
            image_batches = [None] * len(batches)
        assert len(image_batches) == len(batches)

        requests = [
            self._prepare(msgs, imgs)
            for msgs, imgs in zip(batches, image_batches)
        ]
        outputs = engine.generate(requests, params, use_tqdm=False)
        results: List[str] = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            if strip_think:
                text, _ = strip_thinking(text)
            results.append(text.strip())
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HTTP-pool text client (failsafe for large batches)
# ---------------------------------------------------------------------------


class APIPoolTextClient(BaseLLM):
    """Drop-in replacement for VLLMTextClient that dispatches to a pool of
    OpenAI-compatible vLLM HTTP servers.

    Compared to an in-process ``vllm.LLM`` with tensor_parallel_size=8, a pool
    of 8 × tp=1 servers has been observed to be dramatically more stable on
    this hardware (no mysterious ~2-minute process deaths). Each server runs
    in its own Python process and owns exactly one GPU, so NCCL is unused and
    there is no shared broadcast ring that can silently hang.

    The class implements ``generate_batch`` with the same signature as
    :class:`VLLMTextClient`, letting downstream code (BatchInterpreter,
    backfill_rationale) remain unchanged.

    Parameters
    ----------
    api_bases : list of str
        Base URLs like ``["http://localhost:9000/v1", ...]``.
    served_model_name : str
        Name the servers expose (e.g. ``"Qwen/Qwen3-8B"``).
    max_concurrency_per_server : int
        Upper bound on outstanding HTTP requests per server at any time.
    enable_thinking : bool
        Whether to enable Qwen3's thinking mode via
        ``chat_template_kwargs={"enable_thinking": True}``.
    request_timeout : float
        Seconds before we give up on an individual HTTP request.
    """

    def __init__(
        self,
        api_bases: Sequence[str],
        served_model_name: str = "Qwen/Qwen3-8B",
        api_key: str = "EMPTY",
        max_concurrency_per_server: int = 16,
        enable_thinking: bool = True,
        request_timeout: float = 600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(served_model_name, **kwargs)
        if not api_bases:
            raise ValueError("APIPoolTextClient requires at least one api_base")
        self.api_bases: List[str] = list(api_bases)
        self.served_model_name = served_model_name
        self.api_key = api_key
        self.max_concurrency_per_server = max_concurrency_per_server
        self.enable_thinking = enable_thinking
        self.request_timeout = request_timeout
        self._clients: List[Any] = []  # one openai.OpenAI per server

    # ----- housekeeping -----

    def _init_clients(self) -> None:
        if self._clients:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai>=1.0") from e
        for base in self.api_bases:
            self._clients.append(
                OpenAI(api_key=self.api_key, base_url=base,
                       timeout=self.request_timeout)
            )
        logger.info("APIPoolTextClient initialised with %d endpoints",
                    len(self._clients))

    # ----- single-request API (rarely used, just for interface compat) -----

    def generate(self, messages, config=None) -> str:
        self._init_clients()
        outs = self.generate_batch([messages], config)
        return outs[0] if outs else ""

    def generate_with_image(self, messages, image_path, config=None):  # noqa: D401
        logger.warning("APIPoolTextClient does not support images.")
        return self.generate(messages, config)

    # ----- batched API (the real workhorse) -----

    def generate_batch(
        self,
        batches: Sequence[Sequence[Message | Dict]],
        config: Optional[GenerationConfig] = None,
        *,
        adapter: Optional[str | Sequence[str | None]] = None,
        strip_think: bool = True,
        return_thinking: bool = False,
    ) -> List[str] | List[tuple[str, str]]:
        """Dispatch ``batches`` across the server pool with round-robin
        sharding and per-server concurrent requests.

        ``adapter`` may be:
          * ``None``   -> use ``served_model_name`` (base model)
          * ``str``    -> route every payload to that LoRA adapter (must be
                          pre-mounted on every server via ``--lora-modules``)
          * sequence   -> per-payload adapter name (or ``None`` for base);
                          length must equal ``len(batches)``

        Returns a list in the same order as ``batches``.
        """
        self._init_clients()

        # Resolve per-payload model name
        n_payloads = len(batches)
        if adapter is None:
            model_per_payload: List[str] = [self.served_model_name] * n_payloads
        elif isinstance(adapter, str):
            model_per_payload = [adapter] * n_payloads
        else:
            adapter_list = list(adapter)
            if len(adapter_list) != n_payloads:
                raise ValueError(
                    f"adapter sequence length {len(adapter_list)} != "
                    f"batch length {n_payloads}")
            model_per_payload = [
                (a if a else self.served_model_name) for a in adapter_list
            ]

        cfg = config or GenerationConfig()
        max_tokens = cfg.max_tokens if cfg.max_tokens is not None else 2048

        # Prepare payloads
        payloads: List[Dict] = []
        for i, msgs in enumerate(batches):
            msg_dicts = _messages_to_dicts(msgs)
            payload = {
                "model": model_per_payload[i],
                "messages": msg_dicts,
                "temperature": max(cfg.temperature, 0.0),
                "top_p": cfg.top_p,
                "max_tokens": max_tokens,
            }
            if cfg.repetition_penalty is not None:
                payload["extra_body"] = payload.get("extra_body", {})
                payload["extra_body"]["repetition_penalty"] = cfg.repetition_penalty
            # Qwen3 thinking toggle
            ct_kwargs = {"enable_thinking": bool(self.enable_thinking)}
            payload.setdefault("extra_body", {})
            payload["extra_body"]["chat_template_kwargs"] = ct_kwargs
            if cfg.stop_sequences:
                payload["stop"] = list(cfg.stop_sequences)
            payloads.append(payload)

        # Assign shard index per payload (round-robin)
        n_servers = len(self._clients)
        shard_assignments = [i % n_servers for i in range(len(payloads))]

        # Worker function
        import concurrent.futures as cf

        def _call_one(idx: int) -> tuple[int, str]:
            shard = shard_assignments[idx]
            client = self._clients[shard]
            p = payloads[idx]
            try:
                resp = client.chat.completions.create(**p)
                text = resp.choices[0].message.content or ""
            except Exception as e:  # pragma: no cover - surface for debugging
                logger.exception("APIPool request %d (shard %d) failed: %s",
                                 idx, shard, e)
                text = ""
            return idx, text

        total_workers = n_servers * self.max_concurrency_per_server
        logger.info(
            "APIPool dispatch: %d prompts across %d servers "
            "(%d concurrent per server = %d total)",
            len(payloads), n_servers, self.max_concurrency_per_server,
            total_workers,
        )

        results: List[str] = [""] * len(payloads)
        with cf.ThreadPoolExecutor(max_workers=total_workers) as pool:
            for idx, text in pool.map(_call_one, range(len(payloads))):
                results[idx] = text

        # Post-process (strip <think>)
        final: List[Any] = []
        for text in results:
            if strip_think:
                ans, think = strip_thinking(text)
            else:
                ans, think = text, ""
            if return_thinking:
                final.append((ans, think))
            else:
                final.append(ans)
        return final

    def chat_batch(
        self,
        user_messages: Sequence[str],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        *,
        adapter: Optional[str] = None,
        return_thinking: bool = False,
    ) -> List[str] | List[tuple[str, str]]:
        batches = []
        for user in user_messages:
            msgs: List[Dict] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user})
            batches.append(msgs)
        return self.generate_batch(batches, config, adapter=adapter,
                                   return_thinking=return_thinking)


def create_vllm_client(
    model_config,
    *,
    enable_lora: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 16384,
) -> BaseLLM:
    """Return a :class:`VLLMTextClient` or :class:`VLLMVisionClient` from a
    :class:`genpt.config.ModelConfig`.
    """
    model_path = model_config.model_path or model_config.model_name
    if model_config.is_multimodal:
        return VLLMVisionClient(
            model_name=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_thinking=model_config.enable_thinking,
        )
    return VLLMTextClient(
        model_name=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_thinking=model_config.enable_thinking,
        enable_lora=enable_lora,
    )
