"""
Module: Cleaning.py
Description: Handles the Cleaning step of the pipeline.
Author: Diya
Date: <Date>
"""

from pipeline_context import PipelineContext

class CleaningModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        """
        Initialize the module with shared context and optional LLM agent.
        """
        self.context = context
        self.llm_agent = llm_agent

    def run(self, **kwargs) -> bool:
        """
        Execute Cleaning logic.
        - Input: kwargs (parameters for this stage, e.g., missing value strategy, duplicate handling)
        - Output: Updates context, returns True if successful, False otherwise
        """
        self.context.log("Starting Cleaning Module...")

        try:
            # ===============================
            # 1. Load required data from context
            # ===============================
            df = getattr(self.context, "raw_data", None)

            if df is None:
                raise ValueError("No data found in context. Run Ingestion first.")

            # ===============================
            # 2. Apply transformations / logic
            # ===============================
            # Handle missing values
            missing_strategy = kwargs.get("missing_strategy", "mean")  # mean | median | drop
            if df.isna().sum().any():
                if missing_strategy == "mean":
                    df = df.fillna(df.mean(numeric_only=True))
                    self.context.log(" Missing values filled with mean.")
                elif missing_strategy == "median":
                    df = df.fillna(df.median(numeric_only=True))
                    self.context.log(" Missing values filled with median.")
                elif missing_strategy == "drop":
                    df = df.dropna()
                    self.context.log(" Rows with missing values dropped.")
                else:
                    self.context.log(" Unknown missing value strategy, skipped handling.")

            # Handle duplicates
            if kwargs.get("drop_duplicates", True):
                before = len(df)
                df = df.drop_duplicates()
                after = len(df)
                self.context.log(f" Removed {before - after} duplicate rows.")

            # Optional type casting
            dtype_map = kwargs.get("dtype_map", None)  # {col: type}
            if dtype_map:
                df = df.astype(dtype_map, errors="ignore")
                self.context.log(" Applied dtype conversions where possible.")

            # ===============================
            # 3. Optionally use LLM for suggestions
            # ===============================
            if self.llm_agent:
                suggestion = self.llm_agent.ask(
                    f"Stage: Cleaning, Columns: {list(df.columns)}. "
                    f"Data types: {df.dtypes.to_dict()}. "
                    f"Suggest further cleaning steps."
                )
                self.context.log(f" LLM Suggestion: {suggestion}")

            # ===============================
            # 4. Save output back into context
            # ===============================
            setattr(self.context, "cleaned_data", df)
            self.context.status["cleaning"] = "completed"
            self.context.log(" Cleaning completed successfully.")

            # ===============================
            # 5. Optional human approval
            # ===============================
            if not self.context.request_approval(
                "Cleaning", "Proceed with the cleaned dataset?"
            ):
                self.context.status["cleaning"] = "stopped"
                self.context.log("Cleaning stopped by user.")
                return False

            return True

        except Exception as e:
            self.context.status["cleaning"] = "failed"
            self.context.log(f" Cleaning failed: {e}")
            return False
