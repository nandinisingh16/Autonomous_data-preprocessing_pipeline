"""
Module: pipeline_orchestrator.py
Description: Orchestrates all modules in the autonomous data preprocessing pipeline.
Author: Raj Nandini
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pipeline_context import PipelineContext
from ingestion import IngestionModule
from cleaning import CleaningModule
from transformation import TransformationModule
from feature_engineering import FeatureEngineeringModule
from Eda import EDAModule
from TTSplit import TrainTestSplitModule
from vectorization import VectorizationModule
from llm_agent import LLMAgent
from metadata_tracker import MetadataTracker

class PipelineOrchestrator:
    def __init__(self):
        self.context = PipelineContext(stage_name="agentic_pipeline")

        # ⚙️ Smart LLM selection: prefer Transformers (free + local + stable)
        try:
            from transformers import pipeline  # Test if installed
            print("[ℹ️] Using Hugging Face Transformers backend.")
            self.agent = LLMAgent(mode="transformers", model="google/flan-t5-small")
        except ImportError:
            print("[⚠️] Transformers not installed, falling back to Ollama.")
            self.agent = LLMAgent(mode="ollama", model="phi3:mini")

        self.tracker = MetadataTracker()

        # Initialize modules
        self.ingestion = IngestionModule(self.context, self.agent)
        self.cleaning = CleaningModule(self.context, self.agent)

        # Transformation and Feature Engineering modules
        self.transformation = TransformationModule(raw_data=None, llm_agent=self.agent)
        self.feature_engineering = FeatureEngineeringModule(raw_data=None, llm_agent=self.agent)

        self.eda = EDAModule(self.context, self.agent)
        self.ttsplit = TrainTestSplitModule(self.context)
        self.vectorization = VectorizationModule(self.context, self.agent)


    def run_transformation(self):
        """
        Run the transformation module using context.cleaned_data.
        Automatically detects text columns.
        Returns True on success, False otherwise.
        """
        self.context.log(" ► Running Transformation stage...")
        try:
            if getattr(self.context, "cleaned_data", None) is None:
                self.context.log(" No cleaned_data available for transformation.")
                self.context.status["transformation"] = "skipped"
                return True  # not fatal; skip if no cleaned_data

            # Provide data to the module and detect a text column
            self.transformation.raw_data = self.context.cleaned_data.copy()
            text_columns = [col for col in self.transformation.raw_data.select_dtypes(include=["object"]).columns 
                            if self.transformation.raw_data[col].nunique() > 5]
            text_column = text_columns[0] if text_columns else None


            if text_column:
                self.context.log(f" Detected text column for transformation: '{text_column}'")
            else:
                self.context.log(" No text column detected. Skipping text-based transformation.")

            transformed = self.transformation.run(text_column=text_column, apply_balancing=False)

            # Accept either DataFrame return or attribute
            if transformed is not None:
                self.context.transformed_data = transformed
            else:
                self.context.transformed_data = getattr(self.transformation, "transformed_data", None)

            if self.context.transformed_data is None:
                raise RuntimeError("Transformation produced no transformed_data.")

            self.context.status["transformation"] = "completed"
            self.context.log(" Transformation stage completed.")
            return True
        except Exception as e:
            self.context.status["transformation"] = "failed"
            self.context.log(f" Transformation stage failed: {e}")
            return False

   
    def _validate_df(self, df: Optional[pd.DataFrame], stage: str) -> bool:
        """Helper to validate DataFrames between pipeline stages."""
        if df is None:
            self.context.log(f" {stage}: Got None instead of DataFrame")
            return False
        if not isinstance(df, pd.DataFrame):
            self.context.log(f" {stage}: Got {type(df)} instead of DataFrame")
            return False
        if len(df.index) == 0:
            self.context.log(f" {stage}: DataFrame is empty")
            return False
        return True

    def run_feature_engineering(self) -> bool:
        """Run feature engineering stage."""
        self.context.log(" ► Running Feature Engineering stage...")
        try:
            # Get source data with validation
            base_df = None
            if hasattr(self.context, "transformed_data"):
                base_df = self.context.transformed_data
            elif hasattr(self.context, "cleaned_data"):
                base_df = self.context.cleaned_data

            if base_df is None or not isinstance(base_df, pd.DataFrame):
                self.context.log(" No valid DataFrame available for feature engineering")
                self.context.status["feature_engineering"] = "skipped"
                return True

            # Convert numpy array to DataFrame if needed
            if not isinstance(base_df, pd.DataFrame):
                try:
                    base_df = pd.DataFrame(base_df)
                except Exception as e:
                    self.context.log(f" Failed to convert input to DataFrame: {str(e)}")
                    self.context.status["feature_engineering"] = "failed"
                    return False

            # Convert any numpy arrays in DataFrame cells to lists
            for col in base_df.columns:
                if base_df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                    base_df[col] = base_df[col].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )
                    self.context.log(f" Converted numpy arrays to lists in column: {col}")

            # Set raw data for feature engineering
            self.feature_engineering.raw_data = base_df.copy()

            # Auto-detect text column - handle numpy columns safely
            # Debug start
            self.context.log(f"🔍 Debug: Starting text column detection")
            self.context.log(f"🔍 Debug: Columns ({len(base_df.columns)}): {[str(c) for c in base_df.columns[:10]]}")

            text_cols = []
            try:
                text_cols = base_df.select_dtypes(include=["object"]).columns.tolist()
            except Exception as e:
                self.context.log(f"⚠️ select_dtypes failed: {e}")
                self.context.log(f"Base DF dtypes: {base_df.dtypes}")

            text_column = None
            for col in text_cols:
                try:
                    col_name = str(col)
                    if (base_df[col_name].notna().any() and 
                        base_df[col_name].nunique() > 5):
                        text_column = col_name
                        break
                except Exception as e:
                    self.context.log(f"⚠️ Column {col} check failed: {e}")


            # Auto-detect label column - handle numpy columns safely
            label_candidates = {'label', 'target', 'output', 'class', 'survived'}
            label_column = None
            for col in base_df.columns:
                if str(col).lower() in label_candidates:  # Convert to string before comparison
                    label_column = str(col)  # Store as Python str
                    break

            # Use self.context (not undefined 'context') for debug logging
            self.context.log("🔍 Debug: Type of transformed_data: " + str(type(self.context.transformed_data)))
            self.context.log("🔍 Debug: transformed_data columns: " + str(getattr(self.context.transformed_data, 'columns', 'NO COLUMNS')))
            self.context.log("🔍 Debug: Sample transformed_data head:\n" + str(self.context.transformed_data.head() if hasattr(self.context.transformed_data, 'head') else self.context.transformed_data))

            # Run feature engineering with string column names
            engineered = self.feature_engineering.run(
                text_column=text_column,
                label_column=label_column
            )

            # Validate output
            if engineered is None or not isinstance(engineered, pd.DataFrame):
                raise ValueError("Feature engineering returned invalid output")

            # Store results
            self.context.engineered_data = engineered
            self.context.transformed_data = engineered
            self.context.status["feature_engineering"] = "completed"
            self.context.log(" Feature Engineering stage completed.")
            return True

        except Exception as e:
            self.context.status["feature_engineering"] = "failed"
            self.context.log(f" Feature Engineering stage failed: {str(e)}")
            return False

    def run(self, input_file: str = "sample_data.csv"):
        self.context.log(" Starting Agentic Data Preprocessing Pipeline")

        # Ingestion needs file_path
        if not self.ingestion.run(file_path=input_file):
            self.context.log(" Ingestion failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Cleaning (reads context.raw_data and writes context.cleaned_data)
        if not self.cleaning.run():
            self.context.log(" Cleaning failed.")
            self.tracker.record(self.context)
            return self.context.status
            

        # Ensure downstream modules see the cleaned output
        if getattr(self.context, "cleaned_data", None) is None:
            self.context.log(" No cleaned_data available for transformation/EDA.")
            self.context.status["eda"] = "failed"
            self.tracker.record(self.context)
            return self.context.status

        # For now, treat cleaned_data as transformed_data unless another transform stage exists
        self.context.transformed_data = self.context.cleaned_data.copy()

        # Transformation stage
        if not self.run_transformation():
            self.context.log(" Transformation failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Feature Engineering stage
        if not self.run_feature_engineering():
            self.context.log(" Feature Engineering failed.")
            self.tracker.record(self.context)
            return self.context.status

        # EDA (reads context.transformed_data)
        if not self.eda.run():
            self.context.log(" EDA failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Train-Test Split: pass the transformed dataframe
        if not self.ttsplit.run(self.context.transformed_data):
            self.context.log(" Train-Test Split failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Vectorization expects context.split_data with X_train/X_test
        if not self.vectorization.run():
            self.context.log(" Vectorization failed.")
            self.tracker.record(self.context)
            return self.context.status

        self.context.log(" Pipeline completed successfully!")

        # Save metadata after run
        self.tracker.record(self.context)
        self.context.log(" Metadata recorded.")

        return self.context.status

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    orchestrator = PipelineOrchestrator()
    status = orchestrator.run(input_file=input_file)
    print("\nFinal status:", status)
