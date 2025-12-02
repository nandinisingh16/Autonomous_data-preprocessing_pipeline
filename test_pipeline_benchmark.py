"""
Module: test_pipeline_benchmark.py
Description: Benchmark pipeline on multiple datasets
Author: Raj Nandini
Date: 2025-10-28
"""

from datetime import datetime
from llm_agent import create_llm_agent
from pipeline_orchestrator import PipelineOrchestrator
import pandas as pd
import json
import time
import os
import glob
import argparse


def test_pipeline_on_datasets(use_llm=False, llm_provider="openai"):
    """Test pipeline on multiple datasets with mixed LLM settings for comparison."""

    # ✅ GENERIC DATASET LIST - NO TARGET COLUMNS
    test_datasets = [
        "test_datasets/titanic.csv",
        "test_datasets/diabetes.csv",
        "test_datasets/B_cancer.csv",
        "test_datasets/M_cancer.csv"
    ]

    results = []

    # Define LLM configuration per dataset for comparison
    # Some datasets with LLM enabled, some disabled
    dataset_llm_config = {
        "test_datasets/titanic.csv": {"use_llm": True, "provider": llm_provider},
        "test_datasets/diabetes.csv": {"use_llm": False, "provider": "none"},
        "test_datasets/B_cancer.csv": {"use_llm": True, "provider": llm_provider},
        "test_datasets/M_cancer.csv": {"use_llm": False, "provider": "none"}
    }

    print(f"\n🔄 Testing with MIXED LLM configuration:")
    print(f"   • titanic.csv: LLM ENABLED ({llm_provider})")
    print(f"   • diabetes.csv: LLM DISABLED")
    print(f"   • B_cancer.csv: LLM ENABLED ({llm_provider})")
    print(f"   • M_cancer.csv: LLM DISABLED")
    print()

    for file_path in test_datasets:
        if not os.path.exists(file_path):
            print(f"⏭️ Skipping {file_path} (not found)")
            continue
        
        dataset_name = os.path.basename(file_path).replace(".csv", "")
        print(f"\n{'='*80}")
        print(f"📊 Testing: {dataset_name}")
        print(f"File: {file_path}")
        print(f"{'='*80}")
        
        try:
            # Get LLM configuration for this specific dataset
            dataset_config = dataset_llm_config.get(file_path, {"use_llm": False, "provider": "none"})
            dataset_use_llm = dataset_config["use_llm"]
            dataset_provider = dataset_config["provider"]

            # Create LLM agent for this dataset if needed
            if dataset_use_llm:
                print(f"🤖 Using {dataset_provider.upper()} LLM for {dataset_name}")
                dataset_llm_agent = create_llm_agent(use_llm=True, provider=dataset_provider)
            else:
                print(f"🚫 LLM disabled for {dataset_name}")
                dataset_llm_agent = None

            # Load and display dataset info
            df = pd.read_csv(file_path)
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Missing values: {df.isnull().sum().sum()}")
            print(f"Target column: None (unsupervised pipeline)")
            print(f"LLM: {'ENABLED' if dataset_use_llm else 'DISABLED'}")

            # ✅ NO TARGET COLUMN - Generic pipeline
            orchestrator = PipelineOrchestrator(target_col=None, llm_agent=dataset_llm_agent)

            start_time = time.time()
            result = orchestrator.run(input_file=file_path)
            execution_time = time.time() - start_time

            # Store results
            test_result = {
                "dataset_name": dataset_name,
                "file_path": file_path,
                "target_column": None,  # ✅ GENERIC
                "dataset_shape": df.shape,
                "execution_time_seconds": execution_time,
                "ptma_metrics": result.get("autonomy_metrics", {}),
                "llm_enabled": dataset_use_llm,
                "llm_provider": dataset_provider,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            results.append(test_result)
            
            ptma = result.get("autonomy_metrics", {}).get("PTMA", 0)
            print(f"✅ Success | Time: {execution_time:.2f}s | PTMA: {ptma:.3f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "dataset_name": dataset_name,
                "file_path": file_path,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            })
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"pipeline_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: {results_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test autonomous preprocessing pipeline")
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM suggestions")
    parser.add_argument("--llm-provider", 
                       choices=["openai", "anthropic", "groq"], 
                       default="openai", 
                       help="LLM provider")
    args = parser.parse_args()
    
    results = test_pipeline_on_datasets(use_llm=args.with_llm, llm_provider=args.llm_provider)
    print(f"\n📊 Tested {len(results)} datasets with {args.llm_provider.upper()}")