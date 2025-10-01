"""
Module: eda.py
Description: Handles Exploratory Data Analysis (EDA).
Author: Nandini
Date: 2025-10-01
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pipeline_context import PipelineContext

class EDAModule:
    def __init__(self, context: PipelineContext, llm_agent=None):
        self.context = context
        self.llm_agent = llm_agent

    def run(self, target_col: str = None, task_type: str = "classification") -> bool:
        self.context.log("Starting EDA Module...")

        try:
            df = getattr(self.context, "transformed_data", None)
            if df is None:
                raise ValueError("No transformed_data found in context.")

            eda_results = {}

            # 1. Data Overview
            eda_results["shape"] = df.shape
            eda_results["dtypes"] = df.dtypes.astype(str).to_dict()
            eda_results["missing"] = df.isna().sum().to_dict()
            eda_results["memory_usage"] = df.memory_usage().sum()

            # 2. Univariate Analysis
            for col in df.select_dtypes(include=np.number).columns:
                plt.figure()
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Distribution of {col}")
                plt.savefig(f"eda_{col}_hist.png")
                plt.close()

            # 3. Correlation (Numerical)
            corr = df.corr(numeric_only=True)
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", annot=False)
            plt.title("Correlation Heatmap")
            plt.savefig("eda_correlation.png")
            plt.close()
            eda_results["correlation_matrix"] = corr.to_dict()

            # 4. Target Distribution Check
            if target_col and target_col in df.columns:
                target_counts = df[target_col].value_counts(normalize=True).to_dict()
                eda_results["target_distribution"] = target_counts

            # 5. Outlier Detection via Boxplots
            for col in df.select_dtypes(include=np.number).columns:
                plt.figure()
                sns.boxplot(x=df[col])
                plt.title(f"Outlier Check: {col}")
                plt.savefig(f"eda_{col}_box.png")
                plt.close()

            # 6. Feature Importance (Quick Model)
            if target_col and target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[target_col]

                if task_type == "classification":
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)

                model.fit(X, y)
                importances = dict(zip(X.columns, model.feature_importances_))
                eda_results["feature_importance"] = importances

            # Save results
            self.context.eda_results = eda_results
            self.context.status["eda"] = "completed"
            self.context.log("EDA completed successfully.")

            # Optional LLM Suggestion
            if self.llm_agent:
                suggestion = self.llm_agent.ask(
                    f"EDA summary: {eda_results}. "
                    f"Suggest further insights or checks."
                )
                self.context.log(f" LLM Suggestion: {suggestion}")

            return True

        except Exception as e:
            self.context.status["eda"] = "failed"
            self.context.log(f"EDA failed: {e}")
            return False
