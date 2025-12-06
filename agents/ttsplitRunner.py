"""
Runner: ttsplitRunner.py
Description: Executes Train-Test Split module for data preparation.
"""

import os
import sys
import pandas as pd
from pipeline_context import PipelineContext
from ttsplit import TrainTestSplitModule

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ttsplitRunner.py <path_to_transformed_file> [target_col]")
        sys.exit(1)

    transformed_file = sys.argv[1]
    target_col = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(transformed_file):
        print(f"Error: File '{transformed_file}' not found.")
        sys.exit(1)

    context = PipelineContext(stage_name="ttsplit")
    df = pd.read_csv(transformed_file)

    ttsplit = TrainTestSplitModule(context)
    success = ttsplit.run(df, target_col=target_col, stratified=True)

    if success:
        os.makedirs("logs/ttsplit", exist_ok=True)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/ttsplit/ttsplit_{timestamp}.log"

        with open(log_path, "w") as f:
            f.write("\n".join(context.logs))

        print("\n✅ Train-Test Split completed successfully!")
        print(f"Logs saved to: {log_path}")
        print("\nSummary of Splits:")
        for k, v in context.ttsplit_results.items():
            print(f"- {k}: {v}")
    else:
        print("\n Train-Test Split stage failed.")
