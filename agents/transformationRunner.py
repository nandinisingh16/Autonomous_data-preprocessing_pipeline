"""
Runner: transformationRunner.py
Description: Executes the Transformation stage of the pipeline to preprocess and balance the dataset.
"""

import os
import sys
import pandas as pd
from pipeline_context import PipelineContext
from transformation import TransformationModule

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transformationRunner.py <path_to_cleaned_file> [text_column] [apply_balancing(True/False)]")
        sys.exit(1)

    cleaned_file = sys.argv[1]
    text_column = sys.argv[2] if len(sys.argv) > 2 else None
    apply_balancing = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False

    if not os.path.exists(cleaned_file):
        print(f"Error: File '{cleaned_file}' not found.")
        sys.exit(1)

    context = PipelineContext(stage_name="transformation")
    context.cleaned_data = pd.read_csv(cleaned_file)

    transformer = TransformationModule(
        raw_data=context.cleaned_data,
        llm_agent=None  # Replace with your actual LLM agent if needed
    )
