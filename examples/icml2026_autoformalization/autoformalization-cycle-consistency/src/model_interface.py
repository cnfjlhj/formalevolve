"""
Model Interface for Cycle Consistency

Provides abstract interface for LLM operations:
1. Text generation (for formalization)
2. Log probability computation (for cycle consistency scoring)

You can implement your own backend by subclassing LLMInterface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of text generation"""
    text: str
    finish_reason: str = "stop"
    tokens_used: int = 0


@dataclass
class LogProbResult:
    """Result of log probability computation"""
    log_prob: float  # Total log probability
    num_tokens: int  # Number of tokens
    per_token_log_probs: Optional[List[float]] = None  # Per-token breakdown

    @property
    def normalized_log_prob(self) -> float:
        """Log probability normalized by number of tokens"""
        if self.num_tokens == 0:
            return float('-inf')
        return self.log_prob / self.num_tokens


class LLMInterface(ABC):
    """Abstract interface for LLM operations"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[GenerationResult]:
        """
        Generate text completions.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions to generate

        Returns:
            List of GenerationResult objects
        """
        pass

    @abstractmethod
    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        system_prompt: Optional[str] = None,
    ) -> LogProbResult:
        """
        Compute log probability of completion given prompt.

        This is the core operation for cycle consistency scoring.

        Args:
            prompt: The conditioning prompt
            completion: The text whose probability we want to compute
            system_prompt: Optional system prompt

        Returns:
            LogProbResult with log probability information
        """
        pass


class OpenAICompatibleLLM(LLMInterface):
    """
    LLM interface for OpenAI-compatible APIs (OpenAI, vLLM, etc.)

    Requires: openai package
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "dummy",  # Some local servers don't need key
        )
        logger.info(f"Initialized OpenAI-compatible LLM: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[GenerationResult]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

        results = []
        for choice in response.choices:
            results.append(GenerationResult(
                text=choice.message.content,
                finish_reason=choice.finish_reason,
                tokens_used=response.usage.completion_tokens // n if response.usage else 0,
            ))
        return results

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        system_prompt: Optional[str] = None,
    ) -> LogProbResult:
        """
        Compute log probability using the completions API with logprobs.

        Note: This requires the model to support logprobs in the API.
        For chat models, we concatenate and use completion API.
        """
        # Construct full text
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        try:
            # Try using completions API with echo
            response = self.client.completions.create(
                model=self.model,
                prompt=full_prompt + completion,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )

            # Extract logprobs for completion part only
            logprobs = response.choices[0].logprobs
            if logprobs and logprobs.token_logprobs:
                # Find where completion starts.
                #
                # Preferred: use token byte offsets returned by the server to avoid an extra API call.
                prompt_tokens = None
                text_offsets = getattr(logprobs, "text_offset", None)
                if text_offsets:
                    # NOTE: vLLM-style servers often report byte offsets (UTF-8), not Python str indices.
                    boundary = len(full_prompt.encode("utf-8"))
                    for i, off in enumerate(text_offsets):
                        if off is not None and int(off) >= boundary:
                            prompt_tokens = i
                            break
                    if prompt_tokens is None:
                        prompt_tokens = len(logprobs.token_logprobs)
                else:
                    # Fallback: count prompt tokens via a second echo+logprobs request.
                    prompt_tokens = len(
                        self.client.completions.create(
                            model=self.model,
                            prompt=full_prompt,
                            max_tokens=0,
                            echo=True,
                            logprobs=1,
                        ).choices[0].logprobs.tokens
                    )

                completion_logprobs = logprobs.token_logprobs[prompt_tokens:]
                total_log_prob = sum(lp for lp in completion_logprobs if lp is not None)

                return LogProbResult(
                    log_prob=total_log_prob,
                    num_tokens=len(completion_logprobs),
                    per_token_log_probs=completion_logprobs,
                )
        except Exception as e:
            logger.warning(f"Completions API failed: {e}, trying alternative method")

        # Fallback: Use a simple approximation
        # Generate with logprobs and check if it matches
        return self._compute_log_prob_fallback(full_prompt, completion)

    def _compute_log_prob_fallback(self, prompt: str, completion: str) -> LogProbResult:
        """Fallback method when logprobs API is not available"""
        # This is a rough approximation - generate with logprobs
        # and sum the probabilities of the generated tokens
        logger.warning("Using fallback log prob computation (may be less accurate)")

        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=len(completion.split()) * 2,  # rough estimate
                logprobs=1,
                temperature=0,  # greedy to get most likely
            )

            if response.choices[0].logprobs:
                logprobs = response.choices[0].logprobs.token_logprobs
                logprobs = [lp for lp in logprobs if lp is not None]
                return LogProbResult(
                    log_prob=sum(logprobs),
                    num_tokens=len(logprobs),
                    per_token_log_probs=logprobs,
                )
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")

        # Ultimate fallback: return a very low score
        return LogProbResult(log_prob=-100.0, num_tokens=1)


class HuggingFaceLLM(LLMInterface):
    """
    LLM interface using HuggingFace Transformers directly.

    This gives you full control over log probability computation.
    Requires: transformers, torch
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch_dtype if torch_dtype != "auto" else "auto",
        )
        self.model.eval()
        logger.info(f"Model loaded on device: {self.model.device}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[GenerationResult]:
        import torch

        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        results = []
        for _ in range(n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            results.append(GenerationResult(
                text=text,
                finish_reason="stop",
                tokens_used=len(new_tokens),
            ))

        return results

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        system_prompt: Optional[str] = None,
    ) -> LogProbResult:
        """
        Compute exact log probability of completion given prompt.

        This is the key function for cycle consistency!
        """
        import torch
        import torch.nn.functional as F

        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Tokenize prompt and completion separately
        prompt_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids
        full_ids = self.tokenizer(full_prompt + completion, return_tensors="pt").input_ids

        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]

        if full_len <= prompt_len:
            # Completion is empty or tokenization issue
            return LogProbResult(log_prob=0.0, num_tokens=0)

        # Move to device
        full_ids = full_ids.to(self.model.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits

        # Compute log probabilities for completion tokens
        # logits[i] predicts token[i+1], so we shift
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log probs for actual completion tokens
        completion_log_probs = []
        for i in range(prompt_len - 1, full_len - 1):
            next_token_id = full_ids[0, i + 1].item()
            token_log_prob = log_probs[0, i, next_token_id].item()
            completion_log_probs.append(token_log_prob)

        total_log_prob = sum(completion_log_probs)
        num_tokens = len(completion_log_probs)

        return LogProbResult(
            log_prob=total_log_prob,
            num_tokens=num_tokens,
            per_token_log_probs=completion_log_probs,
        )


class DummyLLM(LLMInterface):
    """
    Dummy LLM for testing purposes.
    Returns placeholder values.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[GenerationResult]:
        return [
            GenerationResult(
                text=f"theorem dummy_{i} : True := trivial",
                finish_reason="stop",
                tokens_used=10,
            )
            for i in range(n)
        ]

    def compute_log_prob(
        self,
        prompt: str,
        completion: str,
        system_prompt: Optional[str] = None,
    ) -> LogProbResult:
        import random
        # Return random log prob for testing
        return LogProbResult(
            log_prob=random.uniform(-100, -50),
            num_tokens=len(completion.split()),
        )
