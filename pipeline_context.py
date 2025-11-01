"""
Module: pipeline_context.py
Description: Shared context for the autonomous data preprocessing pipeline.
Author: Team
Date: 2025-09-20
"""

import datetime
import sys

class PipelineContext:
    def __init__(self):
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

    def log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.logs.append(log_message)
        print(log_message)
        sys.stdout.flush()

    def request_approval(self, stage: str, question: str) -> bool:
        self.log(f"Approval requested for {stage}: {question}")
        approved = True  # Auto-approve for now
        if approved:
            self.log(f"Approval granted for {stage}.")
        else:
            self.log(f"Approval denied for {stage}.")
        return approved
