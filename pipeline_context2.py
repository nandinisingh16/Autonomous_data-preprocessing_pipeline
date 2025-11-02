"""
Module: pipeline_context.py
Description: Shared context for managing data, logging, and state across all pipeline stages.
Author: Raj Nandini Singh
Date: 2025-10-28
"""

import os
import json
import datetime
import pandas as pd


class PipelineContext:
    """
    A centralized context object that persists state, data, and logs between pipeline stages.
    """

    def __init__(self, stage_name: str = "init", base_dir: str = "data"):
        self.stage_name = stage_name
        self.base_dir = base_dir

        # --- Stage-wise Directories ---
        self.raw_dir = os.path.join(base_dir, "raw")
        self.cleaned_dir = os.path.join(base_dir, "cleaned")
        self.transformed_dir = os.path.join(base_dir, "transformed")
        self.eda_dir = os.path.join(base_dir, "eda")
        self.split_dir = os.path.join(base_dir, "split")
        self.logs_dir = os.path.join("logs", stage_name)

        # --- Ensure Directories Exist ---
        for d in [
            self.base_dir,
            self.raw_dir,
            self.cleaned_dir,
            self.transformed_dir,
            self.eda_dir,
            self.split_dir,
            self.logs_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        # --- Runtime attributes ---
        self.raw_data = None
        self.cleaned_data = None
        self.transformed_data = None
        self.eda_results = {}
        self.ttsplit_results = {}
        self.vector_store = None

        self.status = {}
        self.logs = []
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.log(f"PipelineContext initialized for stage: {stage_name}")

    # --------------------------
    # Logging Utilities
    # --------------------------
    def log(self, message: str):
        """
        Append message to logs and print to console.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{self.stage_name}] {message}"
        print(log_line)
        self.logs.append(log_line)

    def save_logs(self):
        """
        Persist logs to a file under logs/<stage_name>/timestamp.log
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"{self.stage_name}_{ts}.log")
        with open(log_file, "w") as f:
            f.write("\n".join(self.logs))
        self.log(f"Logs saved to {log_file}")
        return log_file

    # --------------------------
    # Data I/O Utilities
    # --------------------------
    def save_data(self, df: pd.DataFrame, subdir: str, filename: str):
        """
        Save a DataFrame to a CSV file in the pipeline directory.
        """
        path = os.path.join(subdir, filename)
        df.to_csv(path, index=False)
        self.log(f"Data saved to: {path}")
        return path

    def load_data(self, path: str):
        """
        Load CSV data and return a DataFrame.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        self.log(f"Loaded data from: {path} (shape: {df.shape})")
        return df

    # --------------------------
    # State Persistence
    # --------------------------
    def save_context(self):
        """
        Save pipeline state (status + metadata) to JSON for debugging or continuation.
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ctx_file = os.path.join(self.logs_dir, f"context_{ts}.json")
        state = {
            "stage_name": self.stage_name,
            "status": self.status,
            "start_time": self.start_time,
            "eda_results": self.eda_results,
            "ttsplit_results": self.ttsplit_results,
        }
        with open(ctx_file, "w") as f:
            json.dump(state, f, indent=4)
        self.log(f"Pipeline context saved to {ctx_file}")
        return ctx_file

    # --------------------------
    # Lifecycle
    # --------------------------
    def finalize_stage(self, success: bool):
        """
        Called at the end of a stage to persist logs and state.
        """
        self.status[self.stage_name] = "completed" if success else "failed"
        log_path = self.save_logs()
        ctx_path = self.save_context()
        self.log(f"Stage {self.stage_name} finalized. Success: {success}")
        return log_path, ctx_path
