"""
Module: llm_agent.py
Description: Lightweight agent using FREE LLMs (Ollama / Transformers / HuggingFace Free API)
Author: Raj Nandini
Date: 2025-10-28
"""

import os
import json
import subprocess


class LLMAgent:
    def __init__(self, mode="ollama", model="mistral", temperature=0.3):
        """
        mode: 'ollama' | 'transformers' | 'huggingface'
        model: local or remote model name
        """
        self.mode = mode
        self.model = model
        self.temperature = temperature

    def ask(self, prompt: str):
        if self.mode == "ollama":
            return self._ask_ollama(prompt)
        elif self.mode == "transformers":
            return self._ask_transformers(prompt)
        elif self.mode == "huggingface":
            return self._ask_hf(prompt)
        else:
            return "Unsupported mode."

    # ========== LOCAL OLLAMA MODE ==========
    def _ask_ollama(self, prompt):
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=60
            )
            return result.stdout.decode("utf-8").strip()
        except Exception as e:
            return f"Ollama Error: {e}"

    # ========== TRANSFORMERS MODE ==========
    def _ask_transformers(self, prompt):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            pipe = pipeline(
                "text-generation",
                model=self.model or "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype="auto",
                device_map="auto"
            )
            result = pipe(prompt, max_new_tokens=200, temperature=self.temperature)
            return result[0]["generated_text"]
        except Exception as e:
            return f"Transformers Error: {e}"

    # ========== HUGGINGFACE FREE API ==========
    def _ask_hf(self, prompt):
        try:
            import requests

            hf_model = self.model or "tiiuae/falcon-7b-instruct"
            url = f"https://api-inference.huggingface.co/models/{hf_model}"
            headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}

            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
            response = requests.post(url, headers=headers, json=payload)
            data = response.json()
            if isinstance(data, list):
                return data[0].get("generated_text", "")
            return json.dumps(data)
        except Exception as e:
            return f"HuggingFace API Error: {e}"

    # ======== DOMAIN-SPECIFIC HELPERS ========

    def recommend_cleaning(self, summary):
        prompt = (
            "Dataset summary:\n"
            f"{json.dumps(summary, indent=2)}\n\n"
            "Recommend the best missing value handling strategy (drop, mean, median, mode)."
        )
        return self.ask(prompt)

    def suggest_features(self, eda_summary):
        prompt = (
            "Based on the following EDA summary, suggest useful derived features or encodings:\n"
            f"{json.dumps(eda_summary, indent=2)}\n\n"
            "Respond briefly."
        )
        return self.ask(prompt)

    def suggest_next_step(self, pipeline_status):
        prompt = (
            "Pipeline stage status:\n"
            f"{json.dumps(pipeline_status, indent=2)}\n\n"
            "Which preprocessing step should be executed next?"
        )
        return self.ask(prompt)
