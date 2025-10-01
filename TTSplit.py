"""
Module: ttsplit.py
Description: Handles Train-Test Split and Cross-Validation preparation in a modular and agent-friendly way.
Author: Raj Nandini
Date: 2025-10-01
"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
from pipeline_context import PipelineContext  # Your context module

class TrainTestSplitModule:
    def __init__(self, context: PipelineContext):
        self.context = context

    #############################
    # 1. Random Split
    #############################
    def random_split(self, df, target_col=None, test_size=0.2, random_state=42, stratify=False):
        stratify_col = df[target_col] if stratify and target_col else None
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)
        return {"train_shape": train_df.shape, "test_shape": test_df.shape}

    #############################
    # 2. Stratified Split (Classification)
    #############################
    def stratified_split(self, df, target_col, test_size=0.2, random_state=42):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
        return {"train_shape": train_df.shape, "test_shape": test_df.shape, "target_distribution_train": train_df[target_col].value_counts(normalize=True).to_dict(), "target_distribution_test": test_df[target_col].value_counts(normalize=True).to_dict()}

    #############################
    # 3. Time-Series Split
    #############################
    def time_series_split(self, df, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        for train_idx, test_idx in tscv.split(df):
            splits.append({"train_index": train_idx.tolist(), "test_index": test_idx.tolist(),
                           "train_shape": (len(train_idx), df.shape[1]), "test_shape": (len(test_idx), df.shape[1])})
        return splits

    #############################
    # 4. Cross-Validation Prep
    #############################
    def kfold_split(self, df, n_splits=5, target_col=None, stratified=False, groups=None):
        splits = []
        if stratified and target_col:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_gen = cv.split(df, df[target_col])
        elif groups is not None:
            cv = GroupKFold(n_splits=n_splits)
            split_gen = cv.split(df, groups=groups)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_gen = cv.split(df)

        for train_idx, test_idx in split_gen:
            splits.append({"train_index": train_idx.tolist(), "test_index": test_idx.tolist(),
                           "train_shape": (len(train_idx), df.shape[1]), "test_shape": (len(test_idx), df.shape[1])})
        return splits

    #############################
    # 5. Holdout Set
    #############################
    def holdout_set(self, df, holdout_size=0.1, random_state=42):
        train_df, holdout_df = train_test_split(df, test_size=holdout_size, random_state=random_state)
        return {"train_shape": train_df.shape, "holdout_shape": holdout_df.shape}

    #############################
    # 6. Run All Recommended Splits
    #############################
    def run(self, df, target_col=None, test_size=0.2, stratified=False, n_splits=5, holdout_size=0.1):
        self.context.log("Starting Train-Test Split Module...")
        try:
            results = {}
            results["random_split"] = self.random_split(df, target_col=target_col, test_size=test_size, stratify=stratified)
            if stratified and target_col:
                results["stratified_split"] = self.stratified_split(df, target_col=target_col, test_size=test_size)
            results["time_series_split"] = self.time_series_split(df, n_splits=n_splits)
            results["kfold_split"] = self.kfold_split(df, n_splits=n_splits, target_col=target_col, stratified=stratified)
            results["holdout_set"] = self.holdout_set(df, holdout_size=holdout_size)

            self.context.ttsplit_results = results
            self.context.status["ttsplit"] = "completed"
            self.context.log("Train-Test Split completed successfully.")
            return True
        except Exception as e:
            self.context.status["ttsplit"] = "failed"
            self.context.log(f"Train-Test Split failed: {e}")
            return False
