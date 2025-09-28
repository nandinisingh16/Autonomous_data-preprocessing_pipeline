"""
Module: Ingestion.py
Description: Handles the Ingestion step of the pipeline.
Author: Diya
Date: <Date>
"""

from pipeline_context import PipelineContext

class IngestionModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        """
        Initialize the module with shared context and optional LLM agent.
        """
        self.context = context
        self.llm_agent = llm_agent

    def run(self, **kwargs) -> bool:
        """
        Execute Ingestion logic.
        - Input: kwargs (parameters for this stage, e.g., source type, file path, db connection)
        - Output: Updates context, returns True if successful, False otherwise
        """
        self.context.log("Starting Ingestion Module...")

        try:
            # ===============================
            # 1. Load required data from source
            # ===============================
            source_type = kwargs.get("source_type", "csv")
            source_path = kwargs.get("source_path", None)
            df = None

            if source_type == "csv" and source_path:
                import pandas as pd
                df = pd.read_csv(source_path)
                self.context.log(f" Data ingested from CSV: {source_path}")

            elif source_type == "excel" and source_path:
                import pandas as pd
                df = pd.read_excel(source_path)
                self.context.log(f" Data ingested from Excel: {source_path}")

            elif source_type == "database":
                import pandas as pd
                conn_str = kwargs.get("connection_string")
                query = kwargs.get("query", "SELECT * FROM table")
                df = pd.read_sql(query, conn_str)
                self.context.log(f" Data ingested from Database: {query}")

            else:
                raise ValueError("Unsupported ingestion source or missing parameters.")

            # ===============================
            # 2. Basic checks after ingestion
            # ===============================
            if df is None or df.empty:
                raise ValueError("Ingestion produced no data.")

            self.context.log(f" Ingested dataset shape: {df.shape}")

            # ===============================
            # 3. Optionally use LLM for suggestions
            # ===============================
            if self.llm_agent:
                suggestion = self.llm_agent.ask(
                    f"Stage: Ingestion, Columns: {list(df.columns)}. "
                    f"Suggest data quality checks or validation steps."
                )
                self.context.log(f" LLM Suggestion: {suggestion}")

            # ===============================
            # 4. Save output back into context
            # ===============================
            setattr(self.context, "raw_data", df)
            self.context.status["ingestion"] = "completed"
            self.context.log(" Ingestion completed successfully.")

            # ===============================
            # 5. Optional human approval
            # ===============================
            if not self.context.request_approval(
                "Ingestion", "Proceed with the ingested dataset?"
            ):
                self.context.status["ingestion"] = "stopped"
                self.context.log("Ingestion stopped by user.")
                return False

            return True

        except Exception as e:
            self.context.status["ingestion"] = "failed"
            self.context.log(f" Ingestion failed: {e}")
            return False
