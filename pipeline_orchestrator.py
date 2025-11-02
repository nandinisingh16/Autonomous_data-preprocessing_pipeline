"""
Module: pipeline_orchestrator.py
Description: Orchestrates all modules in the autonomous data preprocessing pipeline.
Author: Raj Nandini
Date: 2025-10-28
"""

from pipeline_context import PipelineContext
from ingestion import IngestionModule
from cleaning import CleaningModule
from eda import EDAModule
from ttsplit import TrainTestSplitModule
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

    def run(self):
        self.context.log("🚀 Starting Agentic Data Preprocessing Pipeline")

        if self.ingestion.run():
            if self.cleaning.run():
                if self.eda.run():
                    if self.ttsplit.run():
                        if self.vectorization.run():
                            self.context.log("✅ Pipeline completed successfully!")
                        else:
                            self.context.log("❌ Vectorization failed.")
                    else:
                        self.context.log("❌ Train-Test Split failed.")
                else:
                    self.context.log("❌ EDA failed.")
            else:
                self.context.log("❌ Cleaning failed.")
        else:
            self.context.log("❌ Ingestion failed.")

        # Save metadata after run
        self.tracker.record(self.context)
        self.context.log("📜 Metadata recorded.")

        return self.context.status
