"""
Module: train_test_split.py
Description: Handles splitting data into train/test sets.
Author: Nandini
Date: 2025-10-01
"""

from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from pipeline_context import PipelineContext

class SplitModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        self.context = context
        self.llm_agent = llm_agent

    def run(self, target_col: str, split_type: str = "random", test_size: float = 0.2, n_splits: int = 5):
        self.context.log("Starting Train-Test Split Module...")

        try:
            df = getattr(self.context, "transformed_data", None)
            if df is None:
                raise ValueError("No transformed_data found in context.")

            X = df.drop(columns=[target_col])
            y = df[target_col]

            split_data = {}

            if split_type == "random":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                split_data["train"] = (X_train, y_train)
                split_data["test"] = (X_test, y_test)

            elif split_type == "stratified":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=42
                )
                split_data["train"] = (X_train, y_train)
                split_data["test"] = (X_test, y_test)

            elif split_type == "time_series":
                tscv = TimeSeriesSplit(n_splits=n_splits)
                split_data["folds"] = [
                    ((X.iloc[train_idx], y.iloc[train_idx]), (X.iloc[test_idx], y.iloc[test_idx]))
                    for train_idx, test_idx in tscv.split(X)
                ]

            elif split_type == "kfold":
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                split_data["folds"] = [
                    ((X.iloc[train_idx], y.iloc[train_idx]), (X.iloc[test_idx], y.iloc[test_idx]))
                    for train_idx, test_idx in skf.split(X, y)
                ]

            else:
                raise ValueError(f"Unknown split_type: {split_type}")

            # Save results
            self.context.split_data = split_data
            self.context.status["split"] = "completed"
            self.context.log(f"Train-Test Split ({split_type}) completed successfully.")

            # Optional LLM Suggestion
            if self.llm_agent:
                suggestion = self.llm_agent.ask(
                    f"Split type: {split_type}, Test size: {test_size}, "
                    f"Target distribution: {y.value_counts(normalize=True).to_dict()} "
                    f"Suggest if this split strategy is appropriate."
                )
                self.context.log(f" LLM Suggestion: {suggestion}")

            return True

        except Exception as e:
            self.context.status["split"] = "failed"
            self.context.log(f"Split failed: {e}")
            return False
