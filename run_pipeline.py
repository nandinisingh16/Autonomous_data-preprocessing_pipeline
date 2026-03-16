#!/usr/bin/env python3
"""
Runner script for the Autonomous Data Preprocessing Pipeline
Run this from the project root directory
"""
import sys
import os

# Add current directory to Python path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from orchestrator.pipeline_orchestrator import start_pipeline

if __name__ == "__main__":
    # Get file path from command line argument or use default
    file_path = sys.argv[1] if len(sys.argv) > 1 else "docs/sample-input.csv"
    target_column = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"[INFO] Starting pipeline with file: {file_path}")
    if target_column:
        print(f"[INFO] Target column: {target_column}")

    try:
        result = start_pipeline(input_file=file_path, target_column=target_column)
        print(f"\n  Final Result:\n{result}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)
