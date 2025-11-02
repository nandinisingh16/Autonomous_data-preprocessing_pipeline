"""
Module: metadata_tracker.py
Description: Tracks metadata for each pipeline stage (run history, versions, outcomes).
Author: Raj Nandini
Date: 2025-10-28
"""

import json
import os
import datetime


class MetadataTracker:
    def __init__(self, save_path="metadata/"):
        os.makedirs(save_path, exist_ok=True)
        self.path = os.path.join(save_path, "pipeline_runs.json")
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def record(self, context):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "timestamp": timestamp,
            "status": context.status,
            "logs": context.logs[-5:],  # last few lines only
            "stages": list(context.status.keys()),
        }

        with open(self.path, "r+") as f:
            data = json.load(f)
            data.append(record)
            f.seek(0)
            json.dump(data, f, indent=2)

    def get_history(self, n=5):
        with open(self.path, "r") as f:
            data = json.load(f)
        return data[-n:]
