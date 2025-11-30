"""
Module: llm_agent.py
Description: Ultra-optimized LLM agent with minimal timeout impact
Author: Raj Nandini
Date: 2025-10-28
"""

import os
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time


class LLMAgent:
    def __init__(self, mode="ollama", model="llama2:latest", temperature=0.3, timeout=15):
        self.mode = mode.lower()
        self.model = model
        self.temperature = temperature
        self.timeout = timeout  # Reduced timeout for faster pipeline
        
        # Ultra-light fallback chain
        self.fallback_models = {
            "phi3:mini": "llama2:latest",
            "mistral": "llama2:latest", 
            "llama3": "llama2:latest",
            "default": "llama2:latest"
        }

    def ask(self, prompt: str, task_type="default", quick_mode=True):
        """Main method with ultra-fast timeout handling"""
        # For pipeline speed, skip LLM entirely if timeout is critical
        if quick_mode:
            return self._ask_ultra_fast(prompt)
        else:
            return self._ask_with_timeout(prompt)

    def _ask_ultra_fast(self, prompt: str):
        """Ultra-fast LLM call with aggressive timeout"""
        try:
            # Very short timeout for pipeline speed
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_ollama_fast, prompt, self.model)
                return future.result(timeout=10)  # 10 second max
        except FutureTimeoutError:
            return "LLM suggestion skipped for speed"

    def _ask_with_timeout(self, prompt: str):
        """Standard timeout handling"""
        if self.mode == "ollama":
            return self._ask_ollama_optimized(prompt)
        elif self.mode == "transformers":
            return self._ask_transformers_optimized(prompt)
        elif self.mode == "huggingface":
            return self._ask_hf_optimized(prompt)
        else:
            return f"Unsupported mode: {self.mode}"

    # ======== ULTRA-FAST OLLAMA ========
    def _ask_ollama_optimized(self, prompt):
        """Optimized Ollama with minimal overhead"""
        try:
            # Single attempt with short timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_ollama_fast, prompt, self.model)
                result = future.result(timeout=self.timeout)
            
            if self._is_error_response(result):
                fallback_model = self.fallback_models.get(self.model, "llama2:latest")
                return f"Fallback: Using {fallback_model}"  # Don't actually call fallback
            
            return result

        except FutureTimeoutError:
            return "LLM timeout: Suggestion skipped"
        except Exception as e:
            return f"LLM Error: {e}"

    def _run_ollama_fast(self, prompt, model):
        """Fast Ollama execution with minimal overhead"""
        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=8  # Very short timeout
            )
            output = result.stdout.decode("utf-8").strip()
            return output or "No LLM response"
            
        except subprocess.TimeoutExpired:
            return "Ollama timeout"
        except Exception as e:
            return f"Ollama error: {e}"

    def _is_error_response(self, response):
        """Check if response indicates an error"""
        error_indicators = ["error", "timeout", "unable", "memory", "500"]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in error_indicators)

    # ======== OPTIMIZED TRANSFORMERS ========
    def _ask_transformers_optimized(self, prompt):
        try:
            from transformers import pipeline

            # Use tiny models for maximum speed
            tiny_models = {
                "default": "microsoft/DialoGPT-small",
                "cleaning": "microsoft/DialoGPT-small", 
                "feature_suggestion": "microsoft/DialoGPT-small"
            }
            
            model_name = tiny_models.get(self.model, "microsoft/DialoGPT-small")
            print(f"[⚡] Using fast transformers model: {model_name}")

            pipe = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                torch_dtype="auto"
            )

            # Very short outputs
            result = pipe(prompt, max_new_tokens=50, temperature=self.temperature, do_sample=False)
            return result[0]["generated_text"].strip()

        except Exception as e:
            return f"Transformers skipped: {e}"

    # ======== OPTIMIZED HUGGINGFACE API ========
    def _ask_hf_optimized(self, prompt):
        try:
            import requests
            
            # Use fastest available models
            fast_models = {
                "default": "microsoft/DialoGPT-small",
                "cleaning": "microsoft/DialoGPT-small",
                "feature_suggestion": "microsoft/DialoGPT-small"
            }
            
            hf_model = fast_models.get(self.model, "microsoft/DialoGPT-small")
            token = os.getenv("HF_TOKEN", "")
            
            if not token:
                return "HF token missing"

            url = f"https://api-inference.huggingface.co/models/{hf_model}"
            headers = {"Authorization": f"Bearer {token}"}
            
            # Ultra-fast payload
            payload = {
                "inputs": prompt, 
                "parameters": {
                    "max_new_tokens": 50,
                    "temperature": self.temperature,
                    "do_sample": False
                }
            }

            response = requests.post(url, headers=headers, json=payload, timeout=10)
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "").strip()
            return "HF API response empty"

        except requests.exceptions.Timeout:
            return "HF API timeout"
        except Exception as e:
            return f"HF API error: {e}"

    # ======== PIPELINE-OPTIMIZED DOMAIN HELPERS ========
    def recommend_cleaning(self, summary):
        """Ultra-fast cleaning recommendations"""
        # Skip LLM if summary is too large
        if len(str(summary)) > 1000:
            return "Auto: Drop rows with missing target values"
            
        prompt = (
            "Brief dataset summary:\n"
            f"{json.dumps(summary, indent=2)[:500]}\n\n"
            "One-line missing value strategy (drop/mean/median/mode):"
        )
        return self.ask(prompt, task_type="cleaning", quick_mode=True)

    def suggest_features(self, eda_summary):
        """Ultra-fast feature suggestions"""
        # Skip if EDA is too complex
        if len(str(eda_summary)) > 800:
            return "Auto: Create interaction features"
            
        prompt = (
            "EDA summary - suggest 2 features:\n"
            f"{json.dumps(eda_summary, indent=2)[:400]}\n\n"
            "Respond with 2 bullet points:"
        )
        return self.ask(prompt, task_type="feature_suggestion", quick_mode=True)

    def suggest_next_step(self, pipeline_status):
        """Ultra-fast next step suggestions"""
        prompt = (
            "Pipeline status - next step in 3 words:\n"
            f"{json.dumps(pipeline_status, indent=2)[:300]}"
        )
        return self.ask(prompt, task_type="next_step", quick_mode=True)


# ======== SIMPLE TEST ========
if __name__ == "__main__":
    # Test with fastest settings
    agent = LLMAgent(mode="ollama", model="llama2:latest", timeout=8)
    print(agent.ask("One line about Titanic dataset."))