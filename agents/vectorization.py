"""
Module: vectorization.py
Description: Converts processed data into vectorized numerical representations.
Author: Raj Nandini
Date: 2025-10-28
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from orchestrator.pipeline_context import PipelineContext
from orchestrator.metrics_tracker import metrics


class VectorizationModule:
    def __init__(self, context, llm_agent=None):
        self.context = context  # ✅ ADD THIS
        self.llm_agent = llm_agent
        self.status = {}
        self.logs = []

    def run(self) -> bool:
        """Main vectorization pipeline."""
        try:
            self.log("🎯 Starting Vectorization Module...")

            engineered = getattr(self.context, "engineered_data", None)
            transformed = getattr(self.context, "transformed_data", None)

            if engineered is not None and isinstance(engineered, pd.DataFrame) and not engineered.empty:
                data = engineered
            elif transformed is not None and isinstance(transformed, pd.DataFrame) and not transformed.empty:
                data = transformed
            else:
                data = None

            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.log("❌ No data available for vectorization")
                return False

            self.vectorize_features(data)
            metrics.auto_mod()
            self.log("✅ Features vectorized")

            if self.llm_agent is not None:
                metrics.prompt_used()
                suggestion = self.llm_agent.ask("Best vectorization approach?")
                self.log(f"💡 LLM: {suggestion}")
                metrics.auto_mod()

            self.status["vectorization"] = "completed"
            self.log("✅ Vectorization completed successfully.")
            return True

        except Exception as e:
            self.status["vectorization"] = "failed"
            self.log(f"❌ Vectorization failed: {e}")
            metrics.correction_made()
            return False

    def vectorize_features(self, df):
        """Vectorize features."""
        pass

    def log(self, message: str):
        self.logs.append(message)
        if self.context:
            self.context.log(message)
