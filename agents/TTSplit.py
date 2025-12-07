"""
Module: ttsplit.py
Description: Handles Train-Test Split and Cross-Validation preparation in a modular and agent-friendly way.
Author: Raj Nandini
Date: 2025-10-01
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
from orchestrator.pipeline_context import PipelineContext  # Your context module
from orchestrator.metrics_tracker import metrics
class TrainTestSplitModule:
    def __init__(self, context, llm_agent=None):
        self.context = context  # ✅ ADD THIS
        self.llm_agent = llm_agent
        self.status = {}
        self.logs = []

    #############################
    # 1. Random Split
    #############################
    def random_split(self, df, target_col=None, test_size=0.2, random_state=42, stratify=False):
        stratify_col = df[target_col] if stratify and target_col else None
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)
        metrics.auto_mod()
        return {"train_shape": train_df.shape, "test_shape": test_df.shape}
    

    #############################
    # 2. Stratified Split (Classification)
    #############################
    def stratified_split(self, df, target_col, test_size=0.2, random_state=42):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
        metrics.auto_mod()
        return {"train_shape": train_df.shape, "test_shape": test_df.shape, "target_distribution_train": train_df[target_col].value_counts(normalize=True).to_dict(), "target_distribution_test": test_df[target_col].value_counts(normalize=True).to_dict()}

    #############################
    # 3. Time-Series Split
    #############################
    def time_series_split(self, df, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        metrics.auto_mod()
        for train_idx, test_idx in tscv.split(df):
            splits.append({"train_index": train_idx.tolist(), "test_index": test_idx.tolist(),
                           "train_shape": (len(train_idx), df.shape[1]), "test_shape": (len(test_idx), df.shape[1])})
        return splits

    #############################
    # 4. Cross-Validation Prep
    #############################
    def kfold_split(self, df, n_splits=5, target_col=None, stratified=False, groups=None):
        splits = []
        metrics.auto_mod()
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
        metrics.auto_mod()
        train_df, holdout_df = train_test_split(df, test_size=holdout_size, random_state=random_state)
        return {"train_shape": train_df.shape, "holdout_shape": holdout_df.shape}

    #############################
    # 6. Run All Recommended Splits
    #############################
    def run(self, data: pd.DataFrame, target_col: Optional[str] = None, test_size: float = 0.2, random_state: int = 42) -> bool:
        """
        Execute train-test split.
        
        Args:
            data: Input DataFrame
            target_col: Optional target column for stratified split (if None, random split)
            test_size: Test set proportion
            random_state: Random seed
        """
        self.log("📂 Starting Train-Test Split Module...")
        
        if data is None or data.empty:
            self.log("❌ No data for train-test split")
            return False
        
        try:
            # ✅ IF NO TARGET - Use random split
            if target_col is None or target_col not in data.columns:
                self.log(f"⚠️ No target column specified - using random split")
                from sklearn.model_selection import train_test_split
                X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)
                stratify = None
            else:
                self.log(f"🎯 Using stratified split on '{target_col}'")
                from sklearn.model_selection import train_test_split
                X_train, X_test = train_test_split(
                    data, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=data[target_col]
                )
                stratify = target_col
            
            self.context.train_data = X_train
            self.context.test_data = X_test
            
            self.log(f"✅ Train-Test split completed")
            self.log(f"   Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
            self.status["split"] = "completed"
            metrics.auto_mod()
            return True
            
        except Exception as e:
            self.log(f"❌ Train-Test split failed: {e}")
            self.status["split"] = "failed"
            metrics.correction_made()
            return False

    def split_data(self, df, target_col, stratified):
        """Perform train-test split."""
        pass

    def log(self, message: str):
        self.logs.append(message)
        if self.context:
            self.context.log(message)
