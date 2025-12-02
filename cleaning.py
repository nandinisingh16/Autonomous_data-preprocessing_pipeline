"""
Module: Cleaning.py
Description: Handles the Cleaning step of the pipeline.
Author: Diya
Date: <Date>
"""

import pandas as pd
from pipeline_context import PipelineContext
from metrics_tracker import metrics

class CleaningModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        """
        Initialize the module with shared context and optional LLM agent.
        """
        self.context = context
        self.llm_agent = llm_agent
        self.status = {}

    def run(self, missing_strategy="mean", drop_duplicates=True, dtype_map=None) -> bool:
        """Main cleaning pipeline - SINGLE IMPLEMENTATION."""
        try:
            self.log("🧹 Starting Data Cleaning Module...")

            # Load data
            df = getattr(self.context, "ingested_data", None)
            if df is None:
                self.log("❌ No ingested_data found in context")
                return False

            # Step 1: Handle missing values
            if df.isna().sum().any():
                numeric_cols = df.select_dtypes(include=["number"]).columns
                categorical_cols = df.select_dtypes(exclude=["number"]).columns

                if missing_strategy == "mean":
                    if df[categorical_cols].isna().any().any():
                        df = df.dropna()
                        self.log("⚠️ Dropped rows with missing text data")
                    else:
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                        self.log("✅ Filled numeric missing values with mean")
                        metrics.auto_mod()  # Track auto-modification

            # Step 2: Remove duplicates
            if drop_duplicates:
                before = len(df)
                df = df.drop_duplicates()
                after = len(df)
                if before > after:
                    self.log(f"✅ Removed {before - after} duplicate rows")
                    metrics.auto_mod()  # Track auto-modification

            # Step 3: Apply dtype conversions
            if dtype_map:
                df = df.astype(dtype_map, errors="ignore")
                self.log("✅ Applied dtype conversions")
                metrics.auto_mod()  # Track auto-modification

            # Step 4: LLM suggestions (optional)
            if self.llm_agent:
                self.log("🤖 Consulting LLM for cleaning suggestions...")
                metrics.prompt_used()  # Track LLM call
                suggestion = self.llm_agent.ask(
                    f"Cleaning suggestions for columns: {list(df.columns)}, "
                    f"dtypes: {df.dtypes.to_dict()}"
                )
                self.log(f"💡 LLM: {suggestion}")
                metrics.auto_mod()  # Track LLM recommendation applied

            # Save cleaned data
            self.context.cleaned_data = df
            self.status["cleaning"] = "completed"
            self.log("✅ Data Cleaning completed successfully")
            return True

        except Exception as e:
            self.status["cleaning"] = "failed"
            self.log(f"❌ Cleaning failed: {e}")
            metrics.correction_made()  # Track error correction
            return False

    def log(self, message: str):
        self.context.log(message)
