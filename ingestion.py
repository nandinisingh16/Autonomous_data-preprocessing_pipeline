"""
Module: ingestion.py
Description: Handles the data ingestion step of the pipeline.
Author: Diya Bhandari
Date: 2025-11-01
"""

import pandas as pd
from pipeline_context import PipelineContext
from datetime import datetime
import os

def generate_versioned_filename(base_name: str, stage: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_name = os.path.splitext(os.path.basename(base_name))[0]
    return os.path.join(folder, f"{timestamp}_{clean_name}-{stage}.csv")

class IngestionModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        self.context = context
        self.llm_agent = llm_agent

    def run(self, file_path: str) -> bool:
        self.context.log("Starting Ingestion Module...")

        try:
            #Source identification
            self.context.log(f"Identifying data source: {file_path}")

            #Connection and Access
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

            #Versioning & Lineage tracking (save timestamped copy)
            save_path = generate_versioned_filename(file_path, "ingested", "data/raw")
            df.to_csv(save_path, index=False)
            self.context.log(f"File saved to: {save_path}")

            #Store lineage metadata
            self.context.version_info = {
                "ingested_at": datetime.now().isoformat(),
                "source": file_path,
                "saved_path": save_path
            }

            #Data Validation (basic)
            if df.empty:
                raise ValueError("DataFrame is empty after ingestion.")

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
