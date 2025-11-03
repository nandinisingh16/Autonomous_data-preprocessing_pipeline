"""
Module: feature_engineering.py
Description: Handles comprehensive feature engineering for diverse raw datasets 
             using LLM-based or topic-modeling-based contextual inference.
Author: Ishita Sawhney
Date: 2025-11-02
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


class FeatureEngineeringModule:
    def __init__(self, raw_data=None, llm_agent=None, save_outputs=True, output_dir="feature_outputs"):
        """
        Initializes the FeatureEngineeringModule.
        """
        self.raw_data = raw_data
        self.llm_agent = llm_agent
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.status = {"feature_engineering": "not_started"}
        self.logs = []

    def log(self, message):
        """Utility function for logging progress."""
        self.logs.append(message)
        print(f"[FeatureEngineering] {message}")

    #############################
    # 1. DOMAIN / CONTEXT INFERENCE
    #############################
    def infer_context(self, df, text_col=None):
        """
        Infers dataset context using either an LLM (if available) or unsupervised topic modeling (LDA).
        This step helps guide downstream feature extraction adaptively.
        """
        self.log("Inferring dataset context using LLM or topic modeling...")
        inferred_info = {}

        try:
            if self.llm_agent:
                # Use LLM-based contextual inference
                sample_text = df.sample(min(5, len(df))).to_string(index=False)
                prompt = f"""
                Here is a raw dataset sample:
                {sample_text}
                Infer the general context or type of data this represents and
                suggest meaningful feature extraction ideas (no specific domains).
                """
                inferred_info["context_summary"] = self.llm_agent.ask(prompt)
                self.log(f"LLM-based context inference completed.")
            else:
                # Fallback: Topic Modeling (Unsupervised)
                if text_col and text_col in df.columns:
                    texts = df[text_col].astype(str).tolist()
                    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
                    X = vectorizer.fit_transform(texts)
                    lda = LatentDirichletAllocation(n_components=3, random_state=42)
                    lda.fit(X)
                    words = np.array(vectorizer.get_feature_names_out())
                    topics = [", ".join(words[topic.argsort()[-10:]]) for topic in lda.components_]
                    inferred_info["topics"] = topics
                    self.log(f"Topic modeling inference completed with {len(topics)} topics.")
                else:
                    self.log("No text column provided for topic modeling. Skipping context inference.")
        except Exception as e:
            self.log(f"Context inference failed: {e}")

        return inferred_info

    #############################
    # 2. DOMAIN-INDEPENDENT FEATURE CREATION
    #############################
    def create_domain_independent_features(self, df):
        """
        Creates statistical and structural features that are domain-independent.
        """
        self.log("Creating general domain-independent features...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df["numeric_mean"] = df[numeric_cols].mean(axis=1)
            df["numeric_std"] = df[numeric_cols].std(axis=1)
            df["numeric_sum"] = df[numeric_cols].sum(axis=1)
        df["missing_values"] = df.isnull().sum(axis=1)
        df["non_missing_ratio"] = df.notnull().sum(axis=1) / df.shape[1]
        return df

    #############################
    # 3. POLYNOMIAL AND INTERACTION FEATURES
    #############################
    def polynomial_interaction_features(self, df):
        """
        Generates polynomial and interaction features for all numeric columns.
        """
        self.log("Generating polynomial and interaction features...")
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            self.log("No numeric columns for polynomial feature generation.")
            return df
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        poly_features = poly.fit_transform(numeric_df)
        feature_names = poly.get_feature_names_out(numeric_df.columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names)
        poly_df = poly_df.iloc[:, :min(50, poly_df.shape[1])]  # limit excessive expansion
        df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
        return df

    #############################
    # 4. TEMPORAL FEATURES
    #############################
    def extract_temporal_features(self, df):
        """
        Extracts temporal patterns from datetime-like columns automatically.
        """
        self.log("Extracting temporal features...")
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
                df[f"{col}_hour"] = df[col].dt.hour if hasattr(df[col].dt, "hour") else np.nan
        return df

    #############################
    # 5. TEXT FEATURES
    #############################
    def extract_text_features(self, df):
        """
        Extracts text-based features like word count, character count, and average word length.
        """
        self.log("Extracting text-based features...")
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        for col in text_cols:
            df[f"{col}_word_count"] = df[col].astype(str).apply(lambda x: len(x.split()))
            df[f"{col}_char_count"] = df[col].astype(str).apply(len)
            df[f"{col}_avg_word_len"] = df[f"{col}_char_count"] / (df[f"{col}_word_count"] + 1e-5)
        return df

    #############################
    # 6. CATEGORICAL FEATURE EXTRACTION
    #############################
    def extract_categorical_features(self, df):
        """
        Encodes categorical variables numerically and adds frequency features.
        """
        self.log("Extracting categorical features...")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            freq = df[col].value_counts(normalize=True)
            df[col + "_freq"] = df[col].map(freq)
        return df

    #############################
    # 7. FEATURE SELECTION
    #############################
    def feature_selection(self, df, label_col=None, k=20):
        """
        Selects top K features using ANOVA F-test or chi2 depending on data type.
        """
        self.log("Selecting most informative features...")
        numeric_df = df.select_dtypes(include=[np.number])
        if label_col is None or label_col not in df.columns:
            self.log("No label column provided. Skipping feature selection.")
            return df

        y = df[label_col]
        X = numeric_df.drop(columns=[label_col], errors='ignore')
        if X.shape[1] <= k:
            self.log("Feature count less than k. Skipping feature reduction.")
            return df
        try:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_new = selector.fit_transform(X, y)
            selected_cols = X.columns[selector.get_support()]
            df = df[selected_cols.tolist() + [label_col]]
        except Exception:
            self.log("ANOVA failed. Using chi2 as fallback.")
            selector = SelectKBest(score_func=chi2, k=k)
            X_new = selector.fit_transform(abs(X), y)
            selected_cols = X.columns[selector.get_support()]
            df = df[selected_cols.tolist() + [label_col]]
        return df

    #############################
    # 8. SAVE INTERMEDIATE RESULTS
    #############################
    def save_intermediate(self, df, filename):
        """Saves intermediate DataFrame if save_outputs is True."""
        if self.save_outputs:
            output_path = os.path.join(self.output_dir, filename)
            df.to_csv(output_path, index=False)
            self.log(f"Saved intermediate file: {output_path}")

    #############################
    # 9. MAIN RUN FUNCTION
    #############################
    def run(self, text_column=None, label_column=None):
        """
        Executes the full feature engineering pipeline on a generic dataset.
        """
        self.log("Starting Feature Engineering Module...")
        self.status["feature_engineering"] = "running"

        try:
            if self.raw_data is None:
                raise ValueError("No raw data provided for feature engineering.")

            df = self.raw_data.copy()

            # Step 1: Context Inference
            context_info = self.infer_context(df, text_col=text_column)
            self.save_intermediate(pd.DataFrame.from_dict(context_info, orient='index'), "context_inference.csv")

            # Step 2: Domain-independent feature creation
            df = self.create_domain_independent_features(df)
            self.save_intermediate(df, "step2_domain_independent.csv")

            # Step 3: Polynomial & Interaction features
            df = self.polynomial_interaction_features(df)
            self.save_intermediate(df, "step3_polynomial.csv")

            # Step 4: Temporal features
            df = self.extract_temporal_features(df)
            self.save_intermediate(df, "step4_temporal.csv")

            # Step 5: Text features
            df = self.extract_text_features(df)
            self.save_intermediate(df, "step5_text_features.csv")

            # Step 6: Categorical features
            df = self.extract_categorical_features(df)
            self.save_intermediate(df, "step6_categorical.csv")

            # Step 7: Feature selection
            df = self.feature_selection(df, label_col=label_column)
            self.save_intermediate(df, "step7_selected.csv")

            self.status["feature_engineering"] = "completed"
            self.log("Feature engineering completed successfully.")
            self.engineered_data = df
            return df

        except Exception as e:
            self.status["feature_engineering"] = "failed"
            self.log(f"Feature engineering failed: {e}")
            raise


# ============================
# Example usage (if standalone)
# ============================
if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "text": ["AI transforms industries", "Data drives intelligence", "Learning from data is essential"],
        "timestamp": pd.date_range(start="2025-01-01", periods=3, freq="D"),
        "category": ["A", "B", "A"],
        "value": [10, 20, 15],
        "label": [1, 0, 1]
    })

    fe_module = FeatureEngineeringModule(raw_data=sample_df)
    engineered = fe_module.run(text_column="text", label_column="label")
    print("\nEngineered Data Preview:")
    print(engineered.head())
