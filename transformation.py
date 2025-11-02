"""
Module: transformation.py
Description: Handles comprehensive data transformation for text-based datasets.
Author: Ishita Sawhney
Date: 2025-11-02
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TransformationModule:
    def __init__(self, raw_data=None, llm_agent=None, save_outputs=True, output_dir="transformation_outputs"):
        """
        Initializes the TransformationModule.
        """
        self.raw_data = raw_data
        self.llm_agent = llm_agent
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.status = {"transformation": "not_started"}
        self.logs = []

    def log(self, message):
        """Utility function for logging progress."""
        self.logs.append(message)
        print(f"[Transformation] {message}")

    #############################
    # 1. SCALING / NORMALIZATION
    #############################
    def scale_normalize(self, df, text_col):
        """
        Normalizes text-related numerical features (like word count or sentence length)
        to a common scale using Min-Max Scaling.
        """
        self.log("Performing scaling/normalization on text lengths...")
        df["text_length"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
        df["char_length"] = df[text_col].astype(str).apply(len)
        scaler = MinMaxScaler()
        df[["text_length_scaled", "char_length_scaled"]] = scaler.fit_transform(
            df[["text_length", "char_length"]]
        )
        return df

    #############################
    # 2. ENCODING CATEGORICAL VARIABLES
    #############################
    def encode_categorical(self, df):
        """
        Encodes text-based categorical features using LabelEncoder.
        Automatically detects categorical (non-numeric) columns.
        """
        self.log("Encoding categorical variables...")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    #############################
    # 3. BINNING / DISCRETIZATION
    #############################
    def bin_discretize(self, df):
        """
        Bins continuous text-derived features (like length) into discrete categories.
        """
        self.log("Applying binning/discretization to scaled features...")
        if "text_length_scaled" not in df.columns:
            raise ValueError("Scaling step must be completed before binning.")
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
        df["length_bins"] = discretizer.fit_transform(df[["text_length_scaled"]])
        df["char_bins"] = discretizer.fit_transform(df[["char_length_scaled"]])
        return df

    #############################
    # 4. DIMENSIONALITY REDUCTION
    #############################
    def dimensionality_reduction(self, df):
        """
        Reduces the dimensionality of numeric and encoded text features using Truncated SVD.
        This is particularly helpful for sparse vectorized text data.
        """
        self.log("Performing dimensionality reduction using TruncatedSVD...")
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            self.log("Not enough numeric features for dimensionality reduction. Skipping step.")
            return df

        n_components = min(5, numeric_df.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(numeric_df)
        reduced_df = pd.DataFrame(reduced, columns=[f"svd_comp_{i+1}" for i in range(reduced.shape[1])])
        df = pd.concat([df.reset_index(drop=True), reduced_df], axis=1)
        return df

    #############################
    # 5. BALANCING DATA
    #############################
    def balance_data(self, df):
        """
        Balances imbalanced text datasets using RandomOverSampler.
        Expects a 'label' column for supervised learning tasks.
        """
        self.log("Balancing dataset using RandomOverSampler...")
        if "label" not in df.columns:
            self.log("No 'label' column found. Skipping balancing step.")
            return df

        X = df.drop(columns=["label"])
        y = df["label"]
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        balanced_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="label")], axis=1)
        return balanced_df

    #############################
    # 6. TEXT / SEQUENCE PREPARATION
    #############################
    def text_sequence_preparation(self, df, text_col):
        """
        Tokenizes and pads text sequences for deep learning models.
        Produces fixed-length numeric sequences from raw text.
        """
        self.log("Tokenizing and padding text sequences...")
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        texts = df[text_col].astype(str).tolist()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")
        df["padded_sequences"] = list(padded_sequences)
        df["vocab_size"] = len(tokenizer.word_index) + 1
        return df

    #############################
    # 7. SAVE INTERMEDIATE RESULTS
    #############################
    def save_intermediate(self, df, filename):
        """Saves intermediate processed DataFrame if save_outputs is True."""
        if self.save_outputs:
            output_path = os.path.join(self.output_dir, filename)
            df.to_csv(output_path, index=False)
            self.log(f"Saved intermediate file: {output_path}")

    #############################
    # 8. MAIN RUN FUNCTION
    #############################
    def run(self, text_column, apply_balancing=True):
        """
        Executes the full transformation pipeline for a text dataset.
        """
        self.log("Starting Transformation Module...")
        self.status["transformation"] = "running"

        try:
            if self.raw_data is None:
                raise ValueError("No raw data provided for transformation.")

            df = self.raw_data.copy()
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in dataset.")

            # Step 1: Scaling/Normalization
            df = self.scale_normalize(df, text_column)
            self.save_intermediate(df, "step1_scaled.csv")

            # Step 2: Encoding Categorical
            df = self.encode_categorical(df)
            self.save_intermediate(df, "step2_encoded.csv")

            # Step 3: Binning
            df = self.bin_discretize(df)
            self.save_intermediate(df, "step3_binned.csv")

            # Step 4: Dimensionality Reduction
            df = self.dimensionality_reduction(df)
            self.save_intermediate(df, "step4_reduced.csv")

            # Step 5: Balancing (Optional)
            if apply_balancing:
                df = self.balance_data(df)
                self.save_intermediate(df, "step5_balanced.csv")

            # Step 6: Text Sequence Preparation
            df = self.text_sequence_preparation(df, text_column)
            self.save_intermediate(df, "step6_tokenized.csv")

            # Optional LLM suggestion
            if self.llm_agent:
                suggestion = self.llm_agent.ask(
                    f"Transformation summary columns: {df.columns.tolist()}. "
                    f"Suggest possible improvements for text preprocessing."
                )
                self.log(f"LLM Suggestion: {suggestion}")

            self.status["transformation"] = "completed"
            self.log("Transformation completed successfully.")
            self.transformed_data = df
            return df

        except Exception as e:
            self.status["transformation"] = "failed"
            self.log(f"Transformation failed: {e}")
            raise


# ============================
# Example usage (if standalone)
# ============================
if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "text": ["This is a sample text", "Data preprocessing is essential", "AI models love clean data"],
        "label": [0, 1, 0]
    })
    transformer = TransformationModule(raw_data=sample_df)
    transformed = transformer.run(text_column="text")
    print("\nTransformed Data Preview:")
    print(transformed.head())