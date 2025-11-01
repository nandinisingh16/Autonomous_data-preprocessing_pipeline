"""
Module: ingestion.py
Description: Handles the data ingestion step of the pipeline.
Author: Diya Bhandari
Date: 2025-11-01
"""

import pandas as pd
from pipeline_context import PipelineContext

class IngestionModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        self.context = context
        self.llm_agent = llm_agent

    def run(self, file_path: str) -> bool:
        self.context.log("Starting Ingestion Module...")

        try:
            #Source identification
            self.context.log(f"Identifying data source: {file_path}")

            #Connection and Access (for now, just check existence)
            try:
                open(file_path, "r").close()
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")

            #Data Extraction
            df = pd.read_csv(file_path)
            self.context.log(f"Data extracted successfully with {len(df)} records.")

            #Schema & Metadata capture
            metadata = {
                "columns": list(df.columns),
                "shape": df.shape,
                "missing_values": df.isna().sum().to_dict()
            }
            self.context.log(f"Metadata: {metadata}")

            #Versioning & Lineage tracking (basic timestamp)
            import datetime
            self.context.version_info = {
                "ingested_at": datetime.datetime.now().isoformat(),
                "source": file_path
            }

            #Data Validation (basic)
            if df.empty:
                raise ValueError("DataFrame is empty after ingestion.")

            #Error handling is built into try/except block

            #Landing zone storage (store in context)
            self.context.raw_data = df
            self.context.status["ingestion"] = "completed"
            self.context.log("Ingestion completed successfully.")

            #Optional LLM suggestion
            if self.llm_agent:
                suggestion = self.llm_agent.ask(
                    f"Columns: {list(df.columns)}, Missing counts: {df.isna().sum().to_dict()}. "
                    f"Suggest ingestion checks or quality metrics."
                )
                self.context.log(f"LLM Suggestion: {suggestion}")

            return True

        except Exception as e:
            self.context.status["ingestion"] = "failed"
            self.context.log(f"Ingestion failed: {e}")
            return False
