"""
Runner: featureEngineeringRunner.py
Description: Executes the Feature Engineering stage of the pipeline to generate new features for modeling.
"""

import os
import sys
import pandas as pd
from pipeline_context import PipelineContext
from feature_engineering import FeatureEngineeringModule

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python featureEngineeringRunner.py <path_to_transformed_file> [text_column] [label_column]")
        sys.exit(1)

    transformed_file = sys.argv[1]
    text_column = sys.argv[2] if len(sys.argv) > 2 else None
    label_column = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(transformed_file):
        print(f"Error: File '{transformed_file}' not found.")
        sys.exit(1)

    context = PipelineContext(stage_name="feature_engineering")
    context.transformed_data = pd.read_csv(transformed_file)

    fe_module = FeatureEngineeringModule(
        raw_data=context.transformed_data,
        llm_agent=None  # Replace with actual LLM agent if required
    )

    try:
        engineered_df = fe_module.run(
            text_column=text_column,
            label_column=label_column
        )

        os.makedirs("data/engineered", exist_ok=True)
        os.makedirs("logs/feature_engineering", exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/engineered/engineered_{timestamp}.csv"
        log_path = f"logs/feature_engineering/feature_engineering_{timestamp}.log"

        engineered_df.to_csv(output_path, index=False)
        with open(log_path, "w") as f:
            f.write("\n".join(context.logs))

        print(f"\n✅ Feature Engineering completed successfully!")
        print(f"Engineered file saved to: {output_path}")
        print(f"Logs saved to: {log_path}")

    except Exception as e:
        print(f"\n❌ Feature Engineering stage failed: {e}")
        sys.exit(1)
