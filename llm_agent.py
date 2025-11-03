"""
Module: llm_agent.py
Description: Unified LLM agent supporting multiple backends (Ollama, Transformers, HuggingFace API)
Author: Raj Nandini
Date: 2025-10-28
"""

import os
import json
import subprocess


class LLMAgent:
    def __init__(self, mode="ollama", model="mistral", temperature=0.3):
        self.mode = mode.lower()
        self.model = model
        self.temperature = temperature

        self.fallback_models = {
            "llama3": "phi3:mini",
            "mistral": "phi3:mini",
            "default": "mistral"
        }

    def ask(self, prompt: str):
        if self.mode == "ollama":
            return self._ask_ollama(prompt)
        elif self.mode == "transformers":
            return self._ask_transformers(prompt)
        elif self.mode == "huggingface":
            return self._ask_hf(prompt)
        else:
            return f"Unsupported mode: {self.mode}"

    # ======== LOCAL OLLAMA MODE ========
    def _ask_ollama(self, prompt):
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=90
            )
            output = result.stdout.decode("utf-8").strip()
            error_output = result.stderr.decode("utf-8").strip()

            if (
                "500 Internal Server Error" in error_output
                or "memory" in error_output.lower()
                or "unable to load" in error_output.lower()
            ):
                fallback_model = self.fallback_models.get(self.model, self.fallback_models["default"])
                print(f"[⚠️] Model '{self.model}' too large. Falling back to '{fallback_model}'...")
                result = subprocess.run(
                    ["ollama", "run", fallback_model],
                    input=prompt.encode("utf-8"),
                    capture_output=True,
                    timeout=90
                )
                output = result.stdout.decode("utf-8").strip()

            return output or error_output or "No response from Ollama."

        except FileNotFoundError:
            return "Ollama not found. Please install via 'winget install ollama'."
        except subprocess.TimeoutExpired:
            return "Ollama took too long to respond. Try a smaller model."
        except Exception as e:
            return f"Ollama Error: {e}"

    # ======== TRANSFORMERS MODE ========
    def _ask_transformers(self, prompt):
        try:
            from transformers import pipeline

            model_name = self.model or "google/flan-t5-base"
            print(f"[ℹ️] Using local transformers model: {model_name}")

            # auto-select pipeline type
            if "t5" in model_name.lower() or "flan" in model_name.lower():
                task = "text2text-generation"
            else:
                task = "text-generation"

            pipe = pipeline(
                task,
                model=model_name,
                device_map="auto",
                dtype="auto"
            )

            result = pipe(prompt, max_new_tokens=150, temperature=self.temperature)
            return result[0]["generated_text"].strip()

        except Exception as e:
            return f"Transformers Error: {e}"

    # ======== HUGGINGFACE API MODE ========
    def _ask_hf(self, prompt):
        try:
            import requests
            hf_model = self.model or "tiiuae/falcon-7b-instruct"
            token = os.getenv("HF_TOKEN", "")
            if not token:
                return "Missing HuggingFace token. Set it via environment variable HF_TOKEN."

            url = f"https://api-inference.huggingface.co/models/{hf_model}"
            headers = {"Authorization": f"Bearer {token}"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}

            response = requests.post(url, headers=headers, json=payload)
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "").strip()
            return json.dumps(data, indent=2)

        except Exception as e:
            return f"HuggingFace API Error: {e}"

    # ======== DOMAIN HELPERS ========
    def recommend_cleaning(self, summary):
        prompt = (
            "Dataset summary:\n"
            f"{json.dumps(summary, indent=2)}\n\n"
            "Recommend the best missing value handling strategy (drop, mean, median, mode) briefly."
        )
        return self.ask(prompt)

    def suggest_features(self, eda_summary):
        prompt = (
            "Based on the following EDA summary, suggest useful derived features or encodings:\n"
            f"{json.dumps(eda_summary, indent=2)}\n\n"
            "Respond concisely."
        )
        return self.ask(prompt)

    def suggest_next_step(self, pipeline_status):
        prompt = (
            "Pipeline stage status:\n"
            f"{json.dumps(pipeline_status, indent=2)}\n\n"
            "Which preprocessing step should be executed next?"
        )
        return self.ask(prompt)


# ======== TEST ========
if __name__ == "__main__":
    agent = LLMAgent(mode="transformers", model="google/flan-t5-small")
    print(agent.ask("Give one short line about the Titanic dataset."))
