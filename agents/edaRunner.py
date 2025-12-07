"""
Runner: edaRunner.py
Description: Executes the EDA stage of the pipeline for analysis.
"""

import os
from pipeline_context import PipelineContext
from eda import EDAModule
import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python edaRunner.py <path_to_transformed_file> [target_col]")
        sys.exit(1)

    transformed_file = sys.argv[1]
    target_col = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(transformed_file):
        print(f"Error: File '{transformed_file}' not found.")
        sys.exit(1)

    context = PipelineContext(stage_name="eda")
    context.transformed_data = pd.read_csv(transformed_file)

    eda = EDAModule(context)
    success = eda.run(target_col=target_col)

    if success:
        os.makedirs("data/eda", exist_ok=True)
        os.makedirs("logs/eda", exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = eda.generate_report(context.eda_results, f"data/eda/eda_report_{timestamp}.html")
        log_path = f"logs/eda/eda_{timestamp}.log"

        # Save report and logs
        with open(log_path, "w") as f:
            f.write("\n".join(context.logs))

        print(f"\n EDA completed successfully!")
        print(f"Report saved to: {report_path}")
        print(f"Logs saved to: {log_path}")

    else:
        print("\n EDA stage failed.")
