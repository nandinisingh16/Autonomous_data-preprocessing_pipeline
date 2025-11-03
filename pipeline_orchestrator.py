"""
Module: pipeline_orchestrator.py
Description: Orchestrates all modules in the autonomous data preprocessing pipeline.
Author: Raj Nandini
Date: 2025-10-28
"""

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
        self.agent = LLMAgent(mode="ollama", model="mistral")  # free + local
        self.tracker = MetadataTracker()

        # Initialize modules
        self.ingestion = IngestionModule(self.context, self.agent)
        self.cleaning = CleaningModule(self.context, self.agent)
        self.eda = EDAModule(self.context, self.agent)
        self.ttsplit = TrainTestSplitModule(self.context)
        self.vectorization = VectorizationModule(self.context, self.agent)

    def run(self, input_file: str = "sample_data.csv"):
        self.context.log("🚀 Starting Agentic Data Preprocessing Pipeline")

        # Ingestion needs file_path
        if not self.ingestion.run(file_path=input_file):
            self.context.log("❌ Ingestion failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Cleaning (reads context.raw_data and writes context.cleaned_data)
        if not self.cleaning.run():
            self.context.log("❌ Cleaning failed.")
            self.tracker.record(self.context)
            return self.context.status
            

        # Ensure downstream modules see the cleaned output
        if getattr(self.context, "cleaned_data", None) is None:
            self.context.log("❌ No cleaned_data available for transformation/EDA.")
            self.context.status["eda"] = "failed"
            self.tracker.record(self.context)
            return self.context.status

        # For now, treat cleaned_data as transformed_data unless another transform stage exists
        self.context.transformed_data = self.context.cleaned_data.copy()

        # Transformation stage
        if not self.run_transformation():
            self.context.log("❌ Transformation failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Feature Engineering stage
        if not self.run_feature_engineering():
            self.context.log("❌ Feature Engineering failed.")
            self.tracker.record(self.context)
            return self.context.status

        # EDA (reads context.transformed_data)
        if not self.eda.run():
            self.context.log("❌ EDA failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Train-Test Split: pass the transformed dataframe
        if not self.ttsplit.run(self.context.transformed_data):
            self.context.log("❌ Train-Test Split failed.")
            self.tracker.record(self.context)
            return self.context.status

        # Vectorization expects context.split_data with X_train/X_test
        if not self.vectorization.run():
            self.context.log("❌ Vectorization failed.")
            self.tracker.record(self.context)
            return self.context.status

        self.context.log("✅ Pipeline completed successfully!")

        # Save metadata after run
        self.tracker.record(self.context)
        self.context.log("📜 Metadata recorded.")

        return self.context.status

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    orchestrator = PipelineOrchestrator()
    status = orchestrator.run(input_file=input_file)
    print("\nFinal status:", status)
