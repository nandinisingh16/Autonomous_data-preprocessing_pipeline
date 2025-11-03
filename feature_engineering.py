"""
Module: feature_engineering.py
Description: Robust feature engineering module for tabular datasets.
             - Domain/context inference (LLM or LDA fallback)
             - Domain-independent features (row stats)
             - Polynomial & interaction features (limited expansion)
             - Temporal feature extraction
             - Text-derived features (counts, avg word length)
             - Categorical encoding + frequency features
             - Feature selection (SelectKBest with safe fallbacks)
Author:  Ishita Sawhney
Date: 2025-11-03
"""

from typing import Optional, Dict, Any
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.exceptions import NotFittedError




class FeatureEngineeringModule:
    def __init__(
        self,
        raw_data: Optional[pd.DataFrame] = None,
        llm_agent: Optional[Any] = None,
        save_outputs: bool = True,
        output_dir: str = "feature_outputs",
        max_poly_features: int = 50,
    ):
        """
        Parameters
        ----------
        raw_data: Optional[pd.DataFrame]
            Input dataframe. Can be provided later via .run(df=...).
        llm_agent: Optional[Any]
            Instance providing .ask(prompt) method (optional).
        save_outputs: bool
            Whether to write intermediate CSV outputs to output_dir.
        output_dir: str
            Folder for intermediate outputs.
        max_poly_features: int
            Cap on number of polynomial columns produced to avoid explosion.
        """
        self.raw_data = raw_data
        self.llm_agent = llm_agent
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        self.max_poly_features = max_poly_features

        os.makedirs(self.output_dir, exist_ok=True)

        self.status = {"feature_engineering": "not_started"}
        self.logs = []
        self.engineered_data: Optional[pd.DataFrame] = None

    # ---- Logging helper -------------------------------------------------
    def log(self, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[FeatureEngineering] [{ts}] {message}"
        self.logs.append(line)
        print(line)

    # ---- Utilities -----------------------------------------------------
    def _is_valid_df(self, df: Optional[pd.DataFrame]) -> bool:
        """Explicit DataFrame validation."""
        return (df is not None 
                and isinstance(df, pd.DataFrame) 
                and len(df.index) > 0)

    def _save_df(self, df: pd.DataFrame, name: str):
        if not self.save_outputs:
            return
        if not self._is_valid_df(df):  # Use our validated check
            self.log(f"Invalid DataFrame for saving {name}")
            return
        path = os.path.join(self.output_dir, name)
        try:
            df.to_csv(path, index=False)
            self.log(f"Saved intermediate file: {path}")
        except Exception as e:
            self.log(f"Failed to save {name}: {e}")

    def _validate_input(self, df: Optional[pd.DataFrame], context: str = "") -> bool:
        """Centralized DataFrame validation."""
        if df is None:
            self.log(f"{context}: Input DataFrame is None")
            return False
        if not isinstance(df, pd.DataFrame):
            self.log(f"{context}: Input is not a DataFrame")
            return False
        if len(df.index) == 0:
            self.log(f"{context}: Input DataFrame is empty")
            return False
        return True

    # ---- 1. Context / domain inference --------------------------------
    def infer_context(self, df: pd.DataFrame, text_col: Optional[str] = None) -> Dict[str, Any]:
        """Infer high-level context from the dataset."""
        self.log("Running context inference...")
        info: Dict[str, Any] = {}

        # Validation
        if not self._is_valid_df(df):
            self.log("Invalid DataFrame for context inference.")
            return info

        try:
            if self.llm_agent is not None:
                # Use explicit None check
                sample = df.head(5).to_dict(orient="records")
                prompt = f"Analyze this data sample:\n{sample}"
                resp = self.llm_agent.ask(prompt)
                info["context_summary"] = resp
                self.log("Received LLM context summary.")
                
            elif text_col is not None and text_col in df.columns:
                texts = df[text_col].astype(str).fillna("").tolist()
                if len(texts) > 0:  # Explicit length check
                    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
                    X = vectorizer.fit_transform(texts)
                    if X.shape[1] >= 3:
                        n_topics = 3
                    else:
                        n_topics = 1
                        
                    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                    lda.fit(X)
                    words = vectorizer.get_feature_names_out()
                    topics = []
                    for comp in lda.components_:
                        top_idx = comp.argsort()[-10:][::-1]
                        top_words = [words[i] for i in top_idx]
                        topics.append(top_words)
                    info["topics"] = topics
                    self.log(f"Extracted {len(topics)} LDA topics.")
        except Exception as e:
            self.log(f"Context inference failed: {str(e)}")
    
        return info

    # ---- 2. Domain-independent features --------------------------------
    def create_domain_independent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add row-level aggregate stats and missingness indicators.
        """
        self.log("Creating domain-independent features...")
        if not self._is_valid_df(df):
            self.log("Invalid DataFrame for domain-independent features.")
            return df

        out = df.copy()
        try:
            num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols)>0:
                out["numeric_mean"] = out[num_cols].mean(axis=1)
                out["numeric_std"] = out[num_cols].std(axis=1).fillna(0)
                out["numeric_sum"] = out[num_cols].sum(axis=1)
            out["missing_count"] = out.isnull().sum(axis=1)
            out["non_missing_ratio"] = 1 - (out.isnull().sum(axis=1) / max(1, out.shape[1]))
        except Exception as e:
            self.log(f"Failed domain-independent feature creation: {e}")
        return out

    # ---- 3. Polynomial & interaction features ---------------------------
    def polynomial_interaction_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial + interaction features on numeric columns. Limits output width.
        """
        self.log("Generating polynomial & interaction features...")
        if not self._is_valid_df(df):
            self.log("Invalid DataFrame for polynomial features.")
            return df

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] <= 1:
            self.log("Insufficient numeric columns for polynomial expansion.")
            return df

        try:
            poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
            X_poly = poly.fit_transform(numeric_df.fillna(0))
            feature_names = poly.get_feature_names_out(numeric_df.columns)
            poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)

            # Limit number of polynomial features to avoid explosion
            if poly_df.shape[1] > self.max_poly_features:
                keep_cols = poly_df.columns[: self.max_poly_features]
                poly_df = poly_df[keep_cols]
                self.log(f"Limited polynomial features to first {self.max_poly_features} columns.")

            result = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
            return result
        except Exception as e:
            self.log(f"Polynomial feature generation failed: {e}")
            return df

    # ---- 4. Temporal features ------------------------------------------
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from datetime-like columns: year, month, day, weekday, hour.
        Attempts to coerce columns to datetime where reasonable.
        """
        self.log("Extracting temporal features...")
        if not self._is_valid_df(df):
            self.log("Invalid DataFrame for temporal features.")
            return df

        out = df.copy()
        for col in out.columns:
            try:
                # Skip if already datetime dtype
                if np.issubdtype(out[col].dtype, np.datetime64):
                    dt_col = out[col]
                else:
                    # Heuristic: if column name contains 'date'/'time' or dtype is object and parseable
                    if ("date" in col.lower() or "time" in col.lower()) or out[col].dtype == object:
                        try:
                            dt_col = pd.to_datetime(out[col], errors="coerce")
                        except Exception:
                            continue
                    else:
                        continue

                if dt_col.isna().all():
                    continue

                out[f"{col}_year"] = dt_col.dt.year
                out[f"{col}_month"] = dt_col.dt.month
                out[f"{col}_day"] = dt_col.dt.day
                out[f"{col}_weekday"] = dt_col.dt.weekday
                # hour may not exist for date-only columns
                try:
                    out[f"{col}_hour"] = dt_col.dt.hour
                except Exception:
                    out[f"{col}_hour"] = np.nan
            except Exception:
                continue

        return out

    # ---- 5. Text features ------------------------------------------------
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add simple text metrics for text columns: word count, char count, average word length.
        Detects object/string columns automatically.
        """
        self.log("Extracting text-based features...")
        if not self._is_valid_df(df):
            self.log("Invalid DataFrame for text features.")
            return df

        out = df.copy()
        text_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in text_cols:
            try:
                s = out[col].astype(str).fillna("")
                out[f"{col}_word_count"] = s.apply(lambda x: len(x.split()))
                out[f"{col}_char_count"] = s.apply(len)
                # avoid division by zero
                out[f"{col}_avg_word_len"] = out[f"{col}_char_count"] / (out[f"{col}_word_count"].replace(0, 1))
            except Exception as e:
                self.log(f"Failed text features for {col}: {e}")
        return out

    # ---- 6. Categorical encoding & frequency -----------------------------
    def extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encodes categorical columns and adds frequency features."""
        self.log("Extracting categorical features...")
        if not self._is_valid_df(df):
            self.log("Invalid DataFrame for categorical features.")
            return df

        out = df.copy()
        # Convert column names to strings to avoid numpy.str_ issues
        out.columns = out.columns.astype(str)
        cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
        
        for col in cat_cols:
            try:
                col_str = str(col)  # Ensure string type
                s = out[col_str].astype(str).fillna("missing")
                
                # Label encoding
                le = LabelEncoder()
                out[f"{col_str}_encoded"] = le.fit_transform(s)
                
                # Frequency encoding
                vc = s.value_counts(normalize=True).to_dict()  # Convert to regular Python dict
                out[f"{col_str}_freq"] = s.map(vc).fillna(0.0)
                
            except Exception as e:
                self.log(f"Failed categorical processing for {col_str}: {e}")
            
        return out

    # ---- 7. Feature selection -------------------------------------------
    def feature_selection(self, df: pd.DataFrame, label_col: Optional[str] = None, k: int = 20) -> pd.DataFrame:
        """Select top-k numeric features."""
        self.log("Running feature selection...")
        
        # Input validation
        if not self._is_valid_df(df):
            self.log("Invalid DataFrame for feature selection.")
            return df

        if label_col is None or not isinstance(label_col, str):
            self.log("Invalid label column name.")
            return df
            
        if label_col not in df.columns:
            self.log(f"Label column '{label_col}' not found.")
            return df

        try:
            # Convert all column names to strings to avoid numpy.str_ issues
            df.columns = df.columns.astype(str)
            numeric = df.select_dtypes(include=[np.number])
            
            # Handle numeric vs non-numeric labels
            if label_col in numeric.columns:
                y = numeric[label_col].values  # Convert to numpy array
                X = numeric.drop(columns=[label_col]).values  # Convert to numpy array
                feature_names = [str(col) for col in numeric.drop(columns=[label_col]).columns]
            else:
                y = df[label_col].values  # Convert to numpy array
                X = numeric.values  # Convert to numpy array
                feature_names = [str(col) for col in numeric.columns]

            if X.shape[1] == 0:
                self.log("No numeric features available.")
                return df

            k = min(k, X.shape[1])
            
            # Try ANOVA first
            try:
                selector = SelectKBest(score_func=f_classif, k=k)
                selector.fit(X, y)
                mask = selector.get_support()
                selected_features = [feat for feat, selected in zip(feature_names, mask) if selected]
                
                if label_col in df.columns:
                    return df[[*selected_features, label_col]]
                return df[selected_features]
                
            except Exception as e:
                self.log(f"ANOVA failed: {str(e)}. Trying chi2...")
                
                # Chi2 fallback
                try:
                    X_abs = np.abs(X)
                    selector = SelectKBest(score_func=chi2, k=k)
                    selector.fit(X_abs, y)
                    mask = selector.get_support()
                    selected_features = [feat for feat, selected in zip(feature_names, mask) if selected]
                    
                    if label_col in df.columns:
                        return df[[*selected_features, label_col]]
                    return df[selected_features]
                    
                except Exception as e2:
                    self.log(f"Feature selection failed: {str(e2)}")
                    return df
                    
        except Exception as e:
            self.log(f"Feature selection error: {str(e)}")
            return df

    # ---- 8. Main run pipeline ------------------------------------------
    def run(
        self,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Execute feature engineering pipeline."""
        self.log("Starting Feature Engineering pipeline...")
        self.status["feature_engineering"] = "running"

        try:
            if not self._validate_input(self.raw_data, "run"):
                self.status["feature_engineering"] = "failed"
                return None

            # Make working copy
            result_df = self.raw_data.copy()

            # Convert column names to strings
            result_df.columns = [str(c) for c in result_df.columns]

            # Convert any numpy arrays in cells to lists
            for col in result_df.columns:
                if result_df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                    result_df[col] = result_df[col].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )
                    self.log(f"Converted numpy arrays to lists in column: {col}")

            # Convert list-like cells to hashable tuples where possible; drop column if not convertible.
            cols_to_drop = []
            for col in list(result_df.columns):
                try:
                    has_list = result_df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
                except Exception as e:
                    # defensive: if any cell access raises, mark for drop
                    self.log(f"⚠️ Column {col} check failed: {e}")
                    cols_to_drop.append(col)
                    continue

                if not has_list:
                    continue

                # try converting list/ndarray -> tuple; if fails, mark column to drop
                def _to_hashable_cell(x):
                    if isinstance(x, np.ndarray):
                        try:
                            x = x.tolist()
                        except Exception:
                            raise
                    if isinstance(x, list):
                        # try to convert to tuple (may still contain unhashable members)
                        try:
                            return tuple(x)
                        except Exception:
                            raise
                    return x

                try:
                    result_df[col] = result_df[col].apply(lambda x: _to_hashable_cell(x) if isinstance(x, (list, np.ndarray)) else x)
                    self.log(f"Converted list/ndarray cells to tuples in column: {col}")
                except Exception as e:
                    self.log(f"⚠️ Dropping column '{col}' because it contains non-convertible list-like values: {e}")
                    cols_to_drop.append(col)

            if cols_to_drop:
                result_df = result_df.drop(columns=cols_to_drop)
                self.log(f"Dropped columns with non-hashable list-like cells: {cols_to_drop}")

            # Explicitly drop known sequence/padded columns that are not useful for FE
            if "padded_sequences" in result_df.columns:
                result_df = result_df.drop(columns=["padded_sequences"])
                self.log("Dropped 'padded_sequences' column to avoid unhashable list errors")

            # 🔧 --- SANITY NORMALIZATION BLOCK ---
            result_df.columns = [str(c) for c in result_df.columns]
            for c in result_df.columns:
                result_df[c] = result_df[c].apply(
                    lambda x: x[0] if isinstance(x, (np.ndarray, list)) and len(x) == 1 else x
                )

            
            # Domain-independent features
            result_df = self.create_domain_independent_features(result_df)
            if self._validate_input(result_df, "domain_independent"):
                self._save_df(result_df, "step1_domain.csv")
            
            # Text features if column provided
            if text_column and text_column in result_df.columns:
                result_df = self.extract_text_features(result_df)
                if self._validate_input(result_df, "text"):
                    self._save_df(result_df, "step2_text.csv")

            # Categorical features
            result_df = self.extract_categorical_features(result_df)
            if self._validate_input(result_df, "categorical"):
                self._save_df(result_df, "step3_categorical.csv")

            # Feature selection if label provided
            if label_column and label_column in result_df.columns:
                result_df = self.feature_selection(result_df, label_column)
                if self._validate_input(result_df, "selection"):
                    self._save_df(result_df, "step4_selected.csv")

            # Final validation
            if not self._validate_input(result_df, "final"):
                raise ValueError("Pipeline produced invalid DataFrame")

            self.engineered_data = result_df
            self.status["feature_engineering"] = "completed"
            self.log("Feature engineering completed successfully")
            return result_df

        except Exception as e:
            self.status["feature_engineering"] = "failed"
            self.log(f"Feature engineering failed: {str(e)}")
            return None


# ---------------------------
# Quick standalone demo
# ---------------------------
if __name__ == "__main__":
    demo_df = pd.DataFrame(
        {
            "text": ["AI transforms industries", "Data drives intelligence", "Learning from data is essential"],
            "timestamp": pd.date_range(start="2025-01-01", periods=3, freq="D"),
            "category": ["A", "B", "A"],
            "value": [10, 20, 15],
            "label": [1, 0, 1],
        }
    )

    fem = FeatureEngineeringModule(raw_data=demo_df, save_outputs=True)
    out = fem.run(text_column="text", label_column="label")
    print("\nEngineered Data Preview:")
    print(out.head())
