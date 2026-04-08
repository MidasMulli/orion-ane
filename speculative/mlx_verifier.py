"""
MLX GPU Verifier for Speculative Decoding
Connects to MLX-LM server on port 8899 to verify draft tokens.
"""

import requests
import json
import time

MLX_URL = "http://localhost:8899/v1"
MLX_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"


class MLXVerifier:
    """Verifies draft tokens using the GPU model (Qwen 3.5 9B via MLX)."""

    def __init__(self, url=MLX_URL, model=MLX_MODEL):
        self.url = url
        self.model = model
        self.total_verify_time = 0
        self.total_verify_calls = 0

    def health_check(self):
        """Check if MLX server is running."""
        try:
            r = requests.get(f"{self.url}/models", timeout=5)
            models = r.json().get("data", [])
            model_ids = [m["id"] for m in models]
            if self.model in model_ids:
                return True
            # Try first available model
            if model_ids:
                self.model = model_ids[0]
                return True
            return False
        except Exception:
            return False

    def verify_tokens(self, prompt_text, draft_tokens_text, max_new=1):
        """
        Verify draft tokens by asking the GPU model to continue the prompt.

        Strategy: Send prompt + draft tokens, ask for 1 token.
        Compare the GPU's generation with draft tokens.

        Returns: (accepted_count, verifier_token, verify_time_ms)
        """
        t0 = time.time()

        # Build the full text: prompt + draft tokens
        full_text = prompt_text + draft_tokens_text

        try:
            response = requests.post(
                f"{self.url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": full_text}],
                    "max_tokens": len(draft_tokens_text.split()) + 10,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=30,
            )
            result = response.json()
            verifier_text = result["choices"][0]["message"]["content"]
            verify_time = (time.time() - t0) * 1000
        except Exception as e:
            print(f"  Verifier error: {e}")
            return 0, "", 0

        self.total_verify_time += verify_time
        self.total_verify_calls += 1

        return verifier_text, verify_time

    def generate_continuation(self, prompt, max_tokens=1, temperature=0.0):
        """Simple generation — ask GPU for next tokens (chat format)."""
        t0 = time.time()
        try:
            response = requests.post(
                f"{self.url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=30,
            )
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            gen_time = (time.time() - t0) * 1000
            return text, gen_time
        except Exception as e:
            print(f"  GPU generation error: {e}")
            return "", 0

    def complete_raw(self, prompt_text, max_tokens=10, temperature=0.0):
        """Raw text completion — no chat template. For speculative decode verification."""
        t0 = time.time()
        try:
            response = requests.post(
                f"{self.url}/completions",
                json={
                    "model": self.model,
                    "prompt": prompt_text,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=30,
            )
            result = response.json()
            text = result["choices"][0]["text"]
            gen_time = (time.time() - t0) * 1000
            return text, gen_time
        except Exception as e:
            print(f"  GPU raw completion error: {e}")
            return "", 0

    def verify_draft_sequence(self, messages, draft_text, temperature=0.0):
        """
        Core verification: send prompt expecting the GPU to generate,
        then compare with draft tokens.

        For true spec decode, we'd need token-level logprobs.
        This simplified version compares text prefixes.

        Returns: (n_accepted, verifier_continuation, time_ms)
        """
        t0 = time.time()

        # Ask GPU to generate from the prompt
        try:
            response = requests.post(
                f"{self.url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": len(draft_text) + 20,  # generate enough to compare
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=30,
            )
            result = response.json()
            gpu_text = result["choices"][0]["message"]["content"]
            verify_time = (time.time() - t0) * 1000
        except Exception as e:
            return 0, "", (time.time() - t0) * 1000

        self.total_verify_time += verify_time
        self.total_verify_calls += 1

        # Compare draft vs GPU output word by word
        draft_words = draft_text.split()
        gpu_words = gpu_text.split()

        n_accepted = 0
        for i in range(min(len(draft_words), len(gpu_words))):
            if draft_words[i] == gpu_words[i]:
                n_accepted += 1
            else:
                break

        return n_accepted, gpu_text, verify_time

    @property
    def avg_verify_time(self):
        if self.total_verify_calls == 0:
            return 0
        return self.total_verify_time / self.total_verify_calls
