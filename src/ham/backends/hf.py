"""Real Hugging Face causal-LM backend.

Configurable by model id / device / dtype / quantization. Uses the *exact*
tokenizer for token accounting and ``apply_chat_template`` when available.
Separates prefill vs decode latency. Never auto-downloads large models in tests
(only instantiated when a config explicitly selects ``backend.kind == 'hf'``).
"""

from __future__ import annotations

import time

from ..config import BackendConfig
from .base import Backend, GenerationResult


class HFBackend(Backend):
    kind = "hf"

    def __init__(self, cfg: BackendConfig):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.model_id = cfg.model_id
        self._torch = __import__("torch")

        dtype_map = {
            "float32": self._torch.float32,
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
        }
        if cfg.dtype not in dtype_map:
            raise ValueError(f"unknown dtype {cfg.dtype!r}")
        torch_dtype = dtype_map[cfg.dtype]

        model_kwargs: dict = {"torch_dtype": torch_dtype, "trust_remote_code": cfg.trust_remote_code}
        if cfg.quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # fail loudly, never silently drop quantization
                raise RuntimeError(
                    "quantization requested but bitsandbytes/transformers quant config "
                    "is unavailable; install the [quant] extra."
                ) from exc
            if cfg.quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)

        if cfg.device == "auto":
            self.device = "cuda" if self._torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        if "quantization_config" not in model_kwargs:
            self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            # Cap VRAM at 50% so the shared display GPU (Xorg) has headroom;
            # the NVIDIA Open Kernel Module asserts when VRAM is exhausted.
            torch.cuda.set_per_process_memory_fraction(0.5)

    def supports_cuda_metrics(self) -> bool:
        return self.device == "cuda"

    def _render(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            # Pass enable_thinking=False to disable Qwen3/3.5's <think>…</think>
            # mode so the token budget goes to the answer, not reasoning traces.
            # Jinja templates silently ignore unused kwargs → safe for all models.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        return prompt

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompt: str) -> GenerationResult:
        torch = self._torch
        rendered = self._render(prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        do_sample = self.cfg.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=self.cfg.temperature, top_p=self.cfg.top_p)
            torch.manual_seed(self.cfg.seed)

        # Prefill timing: a single forward pass to build the KV cache.
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model(**inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        prefill = time.perf_counter() - t0

        # Free the prefill KV cache before generate() builds its own — without
        # this, peak VRAM doubles at the transition and the NVIDIA Open Kernel
        # Module on the shared display GPU asserts (NV_ERR_GPU_IN_FULLCHIP_RESET).
        if self.device == "cuda":
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        total_gen = time.perf_counter() - t1

        gen_ids = out[0][prompt_tokens:]
        output_tokens = int(gen_ids.shape[0])
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        decode = max(total_gen - prefill, 0.0)

        # Free generation tensors to prevent VRAM fragmentation over many calls.
        del out
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_latency_s=prefill,
            decode_latency_s=decode,
            total_latency_s=prefill + decode,
            extra={"backend": "hf", "device": self.device, "model_id": self.model_id},
        )
