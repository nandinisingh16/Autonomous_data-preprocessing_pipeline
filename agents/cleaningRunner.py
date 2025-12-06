"""
Runner: cleaningRunner.py
Description: Executes only the cleaning stage of the pipeline for testing.
"""

from pipeline_context import PipelineContext
#from llm_agent import LLMAgent
import sys
import os
import pandas as pd
from cleaning import CleaningModule
from metrics_tracker import metrics

if __name__ == "__main__":
    #Read file path from CLI args
    # Require at least 1 arg (file name)
    if len(sys.argv) < 2:
        print("Usage: python cleaningRunner.py <path_to_raw_file> [missing_strategy]")
        sys.exit(1)
    raw_input = sys.argv[1]
    # Optional 2nd arg: missing value strategy
    missing_strategy = sys.argv[2] if len(sys.argv) > 2 else "mean"

     # Default to data/raw/ if user just gives filename
    if not os.path.isabs(raw_input):
        raw_file_path = os.path.join("data", "raw", raw_input)
        print("Resolved file path:", raw_file_path)

    else:
        raw_file_path = raw_input

    if not os.path.exists(raw_file_path):
        print(f"Error: File '{raw_file_path}' does not exist.")
        sys.exit(1)


    context = PipelineContext()
    context.log(f"Loading raw file for cleaning: {raw_file_path}")
    context.raw_data = pd.read_csv(raw_file_path)

    cleaning = CleaningModule(context)  # add llm_agent when needed

    success = cleaning.run(
        missing_strategy=missing_strategy,
        drop_duplicates=True
    )

    if success:
        os.makedirs("data/cleaned", exist_ok=True)
        os.makedirs("logs/cleaning", exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_path = f"data/cleaned/file-cleaned-{timestamp}.csv"
        log_path = f"logs/cleaning/cleaning_{timestamp}.log"

        context.cleaned_data.to_csv(cleaned_path, index=False)
        with open(log_path, "w") as f:
            f.write("\n".join(context.logs))

        print(f"\nCleaning completed successfully!")
        print(f"Cleaned file saved at: {cleaned_path}")
        print(f"Log file saved at: {log_path}")


        #compare raw and cleaned data, later can separate this functionality into a diff module
        cleaned_df = context.cleaned_data
        raw_df = context.raw_data

        print("=== RAW DATA (Before Cleaning) ===")
        print(raw_df.head(10))  # print top 10 rows

        print("\n=== CLEANED DATA (After Cleaning) ===")
        print(cleaned_df.head(10))

        # show summary of changes
        print("\n=== SUMMARY ===")
        print("Raw shape:", raw_df.shape)
        print("Cleaned shape:", cleaned_df.shape)
        print("Rows removed:", len(raw_df) - len(cleaned_df))
        print("Columns:", raw_df.columns.difference(cleaned_df.columns))

        # Compare only common columns
        common_cols = raw_df.columns.intersection(cleaned_df.columns)

        # Compare only up to the smaller length
        min_len = min(len(raw_df), len(cleaned_df))

        if min_len > 0:
            diff_cells = (raw_df[common_cols].head(min_len).reset_index(drop=True) !=
                          cleaned_df[common_cols].head(min_len).reset_index(drop=True)).sum().sum()
            print("Different cells (within overlapping rows):", diff_cells)
        else:
            print("Different cells: N/A (no overlapping rows)")

    else:
        print("\nCleaning Stage Failed!")

