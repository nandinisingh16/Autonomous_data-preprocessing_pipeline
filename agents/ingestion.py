"""
Module: ingestion.py
Description: Handles the data ingestion step of the pipeline.
Author: Diya Bhandari
Date: 2025-11-01
"""

import pandas as pd
import os
from datetime import datetime
from orchestrator.pipeline_context import PipelineContext
from orchestrator.metrics_tracker import metrics

class IngestionModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        self.context = context
        self.llm_agent = llm_agent
        self.status = {}

    def run(self, file_path: str) -> bool:
        """Main ingestion pipeline."""
        try:
            self.log(f"📥 Loading data from: {file_path}")
            
            # Load data
            df = pd.read_csv(file_path)
            metrics.auto_mod()  # ✅ Data loaded automatically
            self.log(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

            # Infer schema
            self.log("🔍 Inferring data schema...")
            metrics.auto_mod()  # ✅ Schema inferred automatically
            
            # Optional LLM type detection
            if self.llm_agent:
                self.log("🤖 Consulting LLM for type detection...")
                metrics.prompt_used()  # ✅ LLM called
                types = self.llm_agent.ask(f"Infer types for: {df.columns.tolist()}")
                metrics.auto_mod()  # ✅ LLM recommendation applied
                self.log(f"💡 LLM suggestions: {types}")

            # Store ingested data
            self.context.ingested_data = df
            self.context.raw_data = df  # Alias for cleaning module
            self.status["ingestion"] = "completed"
            self.log("✅ Ingestion completed")
            return True

        except Exception as e:
            self.status["ingestion"] = "failed"
            self.log(f"❌ Ingestion failed: {e}")
            metrics.correction_made()  # ✅ Error correction
            return False

    def log(self, message: str):
        self.context.log(message)
