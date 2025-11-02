"""
Module: vectorization.py
Description: Converts processed data into vectorized numerical representations.
Author: Raj Nandini
Date: 2025-10-28
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pipeline_context import PipelineContext


class VectorizationModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        self.context = context
        self.llm_agent = llm_agent

    def run(self):
        self.context.log("Starting Vectorization Module...")

        try:
            df = getattr(self.context, "split_data", None)
            if df is None or "X_train" not in df:
                raise ValueError("Split data missing in context.")

            X_train = df["X_train"]
            X_test = df["X_test"]

            numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

            self.context.log(f"Numeric columns: {list(numeric_cols)}")
            self.context.log(f"Categorical columns: {list(cat_cols)}")

            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
            categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, cat_cols)
                ]
            )

            X_train_vec = preprocessor.fit_transform(X_train)
            X_test_vec = preprocessor.transform(X_test)

            self.context.vectorized_data = {"X_train": X_train_vec, "X_test": X_test_vec}
            self.context.status["vectorization"] = "completed"
            self.context.log("Vectorization completed successfully.")

            return True

        except Exception as e:
            self.context.status["vectorization"] = "failed"
            self.context.log(f"Vectorization failed: {e}")
            return False
