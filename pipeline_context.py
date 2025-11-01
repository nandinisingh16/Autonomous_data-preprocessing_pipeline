"""
Module: pipeline_context.py
Description: Shared context for the autonomous data preprocessing pipeline.
Author: Diya Bhandari
Date: 2025-09-20
"""

import datetime
import sys
import os

class PipelineContext:
    def __init__(self, stage_name="pipeline"):
        # Stage name (helps separate log folders)
        self.stage_name = stage_name

        # Store data at different stages
        self.raw_data = None
        self.cleaned_data = None
        self.transformed_data = None
        self.eda_results = None
        self.split_data = None

        # Track stage status
        self.status = {
            "ingestion": "pending",
            "cleaning": "pending",
            "transformation": "pending",
            "eda": "pending",
            "split": "pending"
        }

        # Centralized logs
        self.logs = []

        # Setup log file path (auto-versioned)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("logs", self.stage_name)
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, f"{timestamp}_{self.stage_name}.log")

    def log(self, message: str):
        """
        Write log messages to both memory and a timestamped file.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # Store in memory
        self.logs.append(log_message)

        # Print to console
        print(log_message)
        sys.stdout.flush()

        # Also write to file
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def request_approval(self, stage: str, question: str) -> bool:
        """
        Placeholder for human approval (auto-approve for now).
        """
        self.log(f"Approval requested for {stage}: {question}")
        approved = True
        if approved:
            self.log(f"Approval granted for {stage}.")
        else:
            self.log(f"Approval denied for {stage}.")
        return approved

