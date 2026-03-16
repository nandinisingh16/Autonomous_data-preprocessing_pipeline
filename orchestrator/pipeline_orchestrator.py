"""
Module: pipeline_orchestrator.py
Description: Orchestrates all modules in the autonomous data preprocessing pipeline.
Author: Raj Nandini
Date: 2025-10-28
"""
import sys
import os
# Add project root to Python path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from orchestrator.metrics_tracker import metrics
from orchestrator.pipeline_context import PipelineContext
from agents.ingestion import IngestionModule
from agents.cleaning import CleaningModule
from agents.transformation import TransformationModule
from agents.feature_engineering import FeatureEngineeringModule
from agents.eda import EDAModule
from agents.TTSplit import TrainTestSplitModule
from agents.vectorization import VectorizationModule
from orchestrator.metadata_tracker import MetadataTracker
import sys
import pandas as pd


class PipelineOrchestrator:
    def __init__(self, target_col=None, llm_agent=None):
        """
        Initialize Pipeline Orchestrator.
        
        Args:
            target_col: Optional target column for supervised learning (can be None)
            llm_agent: Optional LLM agent for suggestions
        """
        self.context = PipelineContext()
        self.ingestion = IngestionModule(self.context, llm_agent=llm_agent)
        self.cleaning = CleaningModule(self.context, llm_agent=llm_agent)
        self.transformation = None
        self.feature_engineering = None
        self.eda = EDAModule(self.context, llm_agent=llm_agent)
        self.ttsplit = TrainTestSplitModule(self.context, llm_agent=llm_agent)
        self.vectorization = VectorizationModule(self.context, llm_agent=llm_agent)
        self.tracker = MetadataTracker()
        self.target_column = target_col  # ✅ CAN BE None
        self.llm_agent = llm_agent

    def run_transformation(self) -> bool:
        """Lazy-init and run TransformationModule."""
        try:
            # Initialize if not already done
            if self.transformation is None:
                self.transformation = TransformationModule(
                    context=self.context,
                    llm_agent=self.llm_agent,
                    save_outputs=True,
                    output_dir="transformation_outputs"
                )
            
            # Get cleaned data
            data = getattr(self.context, "cleaned_data", None)
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.context.log("❌ No cleaned data for transformation")
                return False
            
            # Auto-detect text columns
            text_columns = []
            for col in data.select_dtypes(include=['object', 'string']).columns:
                avg_length = data[col].astype(str).apply(len).mean()
                if avg_length > 20:  # Likely text if average length > 20 chars
                    text_columns.append(col)
            
            self.context.log(f"  Detected text columns: {text_columns or 'None'}")
            
            # Run transformation with appropriate parameters
            result = self.transformation.run(
                data=data,  # Pass data explicitly
                target_column=self.target_column,
                text_columns=text_columns if text_columns else None,
                config={
                    'scaling_method': 'auto',
                    'binning_enabled': True,
                    'balancing_enabled': False,  # Don't balance in transformation stage
                    'text_processing_enabled': bool(text_columns)
                }
            )
            
            if result is False:
                self.context.log("❌ Transformation returned False")
                return False
            
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    self.context.log("❌ Transformation returned empty DataFrame")
                    return False
                self.context.transformed_data = result
                self.context.log(f"✅ Transformation completed. Shape: {result.shape}")
                return True
            else:
                self.context.log(f"❌ Transformation returned unexpected type: {type(result)}")
                return False
                
        except Exception as e:
            self.context.log(f"❌ Transformation error: {e}")
            import traceback
            self.context.log(f"Traceback: {traceback.format_exc()}")
            return False
    def run_feature_engineering(self) -> bool:
        """Lazy-init and run FeatureEngineeringModule."""
        try:
            if self.feature_engineering is None:
                self.feature_engineering = FeatureEngineeringModule(self.context, llm_agent=self.llm_agent)

            result = self.feature_engineering.run()
            # If we get here, feature engineering succeeded
            return True
        except Exception as e:
            self.context.log(f"❌ Feature Engineering failed: {e}")
            return False

    def run(self, input_file: str = "sample_data.csv"):
        """Execute full pipeline with metrics tracking."""
        try:
            self.context.log("  Starting Autonomous Data Preprocessing Pipeline")
            metrics.reset()  # Fresh metrics for this run

            # ============================================
            # STAGE 1: INGESTION
            # ============================================
            self.context.log("▶️ STAGE 1: Data Ingestion")
            metrics.task_executed()
            if not self.ingestion.run(file_path=input_file):
                self.context.log("❌ Ingestion failed")
                metrics.correction_made()
                self.context.status["ingestion"] = "failed"
                return self.context.status
            self.context.status["ingestion"] = "completed"
            self.context.log("✅ Ingestion completed")

            # ============================================
            # STAGE 2: CLEANING
            # ============================================
            self.context.log("▶️ STAGE 2: Data Cleaning")
            metrics.task_executed()
            if not self.cleaning.run(missing_strategy="mean", drop_duplicates=True):
                self.context.log("❌ Cleaning failed")
                metrics.correction_made()
                self.context.status["cleaning"] = "failed"
                return self.context.status
            self.context.status["cleaning"] = "completed"
            self.context.log("✅ Cleaning completed")

            # ============================================
            # STAGE 3: TRANSFORMATION
            # ============================================
            self.context.log("▶️ STAGE 3: Data Transformation")
            metrics.task_executed()
            if not self.run_transformation():
                self.context.log("❌ Transformation failed")
                metrics.correction_made()
                self.context.status["transformation"] = "failed"
                return self.context.status
            self.context.status["transformation"] = "completed"
            self.context.log("✅ Transformation completed")

            # ============================================
            # STAGE 4: FEATURE ENGINEERING
            # ============================================
            self.context.log("▶️ STAGE 4: Feature Engineering")
            metrics.task_executed()
            if not self.run_feature_engineering():
                self.context.log("❌ Feature Engineering failed")
                metrics.correction_made()
                self.context.status["feature_engineering"] = "failed"
                return self.context.status
            self.context.status["feature_engineering"] = "completed"
            self.context.log("✅ Feature Engineering completed")

            
            # STAGE 5: EDA
            # ============================================
            self.context.log("▶️ STAGE 5: Exploratory Data Analysis")
            metrics.task_executed()

            try:
                eda_result = self.eda.run()
                
                # Handle different return types
                if isinstance(eda_result, bool):
                    if not eda_result:  # False means failure
                        self.context.log("❌ EDA failed")
                        metrics.correction_made()
                        self.context.status["eda"] = "failed"
                        return self.context.status
                    # True means success
                    self.context.status["eda"] = "completed"
                    self.context.log("✅ EDA completed")
                elif eda_result is None:
                    self.context.log("❌ EDA failed (returned None)")
                    metrics.correction_made()
                    self.context.status["eda"] = "failed"
                    return self.context.status
                else:
                    # Any other return type (DataFrame, dict, etc.) is considered success
                    self.context.status["eda"] = "completed"
                    self.context.log("✅ EDA completed")
                
            except Exception as e:
                self.context.log(f"❌ EDA error: {e}")
                metrics.correction_made()
                self.context.status["eda"] = "failed"
                return self.context.status

            # ============================================
            # STAGE 6: TRAIN-TEST SPLIT
            # ============================================
            self.context.log("▶️ STAGE 6: Train-Test Split")
            metrics.task_executed()
            if not self.ttsplit.run(self.context.cleaned_data, target_col=self.target_column):
                self.context.log("❌ Train-Test Split failed")
                metrics.correction_made()
                self.context.status["split"] = "failed"
                return self.context.status
            self.context.status["split"] = "completed"
            self.context.log("✅ Train-Test Split completed")

            # ============================================
            # STAGE 7: VECTORIZATION
            # ============================================
            self.context.log("▶️ STAGE 7: Vectorization")
            metrics.task_executed()
            if not self.vectorization.run():
                self.context.log("❌ Vectorization failed")
                metrics.correction_made()
                self.context.status["vectorization"] = "failed"
                return self.context.status
            self.context.status["vectorization"] = "completed"
            self.context.log("✅ Vectorization completed")

            # ============================================
            # PIPELINE COMPLETE
            # ============================================
            final_metrics = metrics.to_dict()
            self.context.status["autonomy_metrics"] = final_metrics
            self.context.log(f"✅ Pipeline completed successfully!")
            self.context.log(f"  PTMA Metrics: {final_metrics}")
            
            self.tracker.record(self.context)
            return self.context.status

        except Exception as e:
            self.context.log(f"  Pipeline error: {e}")
            metrics.correction_made()
            self.context.status["autonomy_metrics"] = metrics.to_dict()
            return self.context.status

def start_pipeline(input_file: str = "sample_data.csv", target_column: str = None, llm_agent=None):
    """Entry point for starting the pipeline."""
    orchestrator = PipelineOrchestrator(target_col=target_column, llm_agent=llm_agent)
    return orchestrator.run(input_file)

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    result = start_pipeline(input_file=file_path)
    print(f"\n  Final Result:\n{result}")
