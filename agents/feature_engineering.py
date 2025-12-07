"""
Module: feature_engineering.py
Description: Generic feature engineering module for any dataset (tabular, text, mixed).
Author: Ishita Sawhney
Date: 2025-11-03
Enhanced: 2025-12-02
"""

from typing import Optional, Dict, Any, List
import os
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn modules with fallbacks
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer, PolynomialFeatures
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available, some features disabled")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from orchestrator.metrics_tracker import metrics


class DatasetAnalyzer:
    """Analyzes dataset to determine feature engineering strategies."""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis.
        
        Returns:
        --------
        dict: Analysis results
        """
        analysis = {
            'shape': df.shape,
            'dtypes': {},
            'column_types': {
                'numeric': [],
                'categorical': [],
                'text': [],
                'datetime': [],
                'binary': [],
                'id_like': []
            },
            'missing_stats': {},
            'unique_counts': {},
            'target_candidates': []
        }
        
        # Analyze each column
        for col in df.columns:
            # Data type
            dtype = str(df[col].dtype)
            analysis['dtypes'][col] = dtype
            
            # Missing values
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            analysis['missing_stats'][col] = {
                'missing': missing,
                'missing_pct': missing_pct
            }
            
            # Unique values
            unique = df[col].nunique()
            analysis['unique_counts'][col] = unique
            
            # Classify column type
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis['column_types']['numeric'].append(col)
                if unique == 2:
                    analysis['column_types']['binary'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['column_types']['datetime'].append(col)
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Check if it's text or categorical
                avg_length = df[col].astype(str).apply(len).mean()
                if unique > 10 and avg_length > 20 and unique > len(df) * 0.1:
                    analysis['column_types']['text'].append(col)
                else:
                    analysis['column_types']['categorical'].append(col)
            
            # Check if column looks like an ID
            if col.lower() in ['id', 'index', 'key', 'uuid', 'guid']:
                analysis['column_types']['id_like'].append(col)
            elif unique == len(df) and pd.api.types.is_string_dtype(df[col]):
                analysis['column_types']['id_like'].append(col)
        
        # Find target candidates
        for col in df.columns:
            unique = analysis['unique_counts'][col]
            # Binary classification candidate
            if unique == 2 and col not in analysis['column_types']['id_like']:
                analysis['target_candidates'].append(('binary', col))
            # Multi-class classification candidate
            elif 3 <= unique <= 20 and col not in analysis['column_types']['id_like']:
                analysis['target_candidates'].append(('multiclass', col))
            # Regression candidate (continuous)
            elif unique > 20 and pd.api.types.is_numeric_dtype(df[col]):
                analysis['target_candidates'].append(('regression', col))
        
        return analysis
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        """
        Intelligently detect target column.
        
        Returns:
        --------
        str or None: Target column name
        """
        # Common target names (case-insensitive)
        common_targets = [
            'target', 'label', 'class', 'outcome', 'response',
            'survived', 'y', 'dependent', 'result', 'diagnosis',
            'churn', 'fraud', 'click', 'conversion', 'default',
            'price', 'salary', 'income', 'value', 'score'
        ]
        
        # Check exact matches
        for col in df.columns:
            if col.lower() in [t.lower() for t in common_targets]:
                return col
        
        # Check for binary columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique = df[col].nunique()
                if unique == 2:
                    return col
        
        # Check last column (common convention)
        if len(df.columns) > 0:
            last_col = df.columns[-1]
            unique = df[last_col].nunique()
            if unique < len(df) * 0.5:  # Not too many unique values
                return last_col
        
        return None


class FeatureEngineeringModule:
    def __init__(self, context=None, llm_agent=None, save_outputs=True, output_dir="feature_outputs"):
        """
        Initialize Feature Engineering Module.
        
        Parameters:
        -----------
        context : PipelineContext
            Pipeline context with data
        llm_agent : LLMAgent, optional
            LLM agent for suggestions
        save_outputs : bool, default=True
            Whether to save intermediate outputs
        output_dir : str, default="feature_outputs"
            Output directory
        """
        self.context = context
        self.llm_agent = llm_agent
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.status = {"feature_engineering": "not_started"}
        self.logs = []
        self.engineered_data = None
        self.analysis = None
        
        # Configuration
        self.config = {
            'text_features': True,
            'datetime_features': True,
            'interaction_features': True,
            'polynomial_features': False,
            'clustering_features': True,
            'dimensionality_reduction': True,
            'feature_selection': True,
            'max_polynomial_degree': 2,
            'max_text_features': 10,
            'max_clusters': 5
        }

    # ---- Logging -------------------------------------------------------
    def log(self, message: str):
        """Log message."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[FeatureEngineering] [{ts}] {message}"
        self.logs.append(line)
        print(line)
        if self.context is not None:
            self.context.log(message)

    # ---- Utilities ----------------------------------------------------
    def _is_valid_df(self, df: Optional[pd.DataFrame]) -> bool:
        """Validate DataFrame."""
        return (df is not None 
                and isinstance(df, pd.DataFrame) 
                and not df.empty)

    def _save_df(self, df: pd.DataFrame, name: str):
        """Save intermediate DataFrame."""
        if not self.save_outputs or not self._is_valid_df(df):
            return
        path = os.path.join(self.output_dir, name)
        try:
            df.to_csv(path, index=False)
            self.log(f"💾 Saved: {path}")
        except Exception as e:
            self.log(f"❌ Failed to save {name}: {e}")

    # ---- 1. Dataset Analysis ------------------------------------------
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        self.log("🔍 Analyzing dataset...")
        analyzer = DatasetAnalyzer()
        analysis = analyzer.analyze(df)
        
        self.log(f"📊 Dataset shape: {analysis['shape']}")
        self.log(f"📈 Column types:")
        for col_type, cols in analysis['column_types'].items():
            if cols:
                self.log(f"  - {col_type}: {len(cols)} columns")
        
        if analysis['target_candidates']:
            self.log(f"🎯 Target candidates:")
            for target_type, col in analysis['target_candidates']:
                self.log(f"  - {col} ({target_type})")
        
        return analysis

    # ---- 2. Basic Statistical Features --------------------------------
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic statistical features."""
        self.log("📊 Creating basic statistical features...")
        
        out = df.copy()
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            try:
                # Row-wise statistics
                out["numeric_mean"] = out[numeric_cols].mean(axis=1)
                out["numeric_std"] = out[numeric_cols].std(axis=1).fillna(0)
                out["numeric_sum"] = out[numeric_cols].sum(axis=1)
                out["numeric_min"] = out[numeric_cols].min(axis=1)
                out["numeric_max"] = out[numeric_cols].max(axis=1)
                out["numeric_range"] = out["numeric_max"] - out["numeric_min"]
                
                # Missing value features
                out["missing_count"] = out.isnull().sum(axis=1)
                out["missing_ratio"] = out["missing_count"] / out.shape[1]
                
                metrics.auto_mod()
            except Exception as e:
                self.log(f"⚠️ Basic features failed: {e}")
        
        return out

    # ---- 3. Text Feature Engineering ----------------------------------
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from text columns."""
        if not self.config['text_features']:
            return df
        
        text_cols = self.analysis['column_types']['text']
        if not text_cols:
            return df
        
        self.log(f"📝 Creating text features from {len(text_cols)} columns...")
        
        out = df.copy()
        
        for col in text_cols[:3]:  # Limit to first 3 text columns
            try:
                s = out[col].astype(str).fillna("")
                
                # Basic text metrics
                out[f"{col}_char_count"] = s.apply(len)
                out[f"{col}_word_count"] = s.apply(lambda x: len(x.split()))
                out[f"{col}_avg_word_length"] = out[f"{col}_char_count"] / out[f"{col}_word_count"].replace(0, 1)
                out[f"{col}_digit_count"] = s.apply(lambda x: sum(c.isdigit() for c in x))
                out[f"{col}_uppercase_count"] = s.apply(lambda x: sum(c.isupper() for c in x))
                out[f"{col}_lowercase_count"] = s.apply(lambda x: sum(c.islower() for c in x))
                out[f"{col}_special_count"] = s.apply(lambda x: sum(not c.isalnum() and not c.isspace() for c in x))
                
                # Sentiment analysis if available
                if TEXTBLOB_AVAILABLE:
                    try:
                        out[f"{col}_sentiment"] = s.apply(lambda x: TextBlob(x).sentiment.polarity)
                        out[f"{col}_subjectivity"] = s.apply(lambda x: TextBlob(x).sentiment.subjectivity)
                    except:
                        pass
                
                # TF-IDF if available and text is not too long
                if SKLEARN_AVAILABLE and s.str.len().mean() < 1000:
                    try:
                        vectorizer = TfidfVectorizer(max_features=self.config['max_text_features'])
                        tfidf_matrix = vectorizer.fit_transform(s)
                        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                                              columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
                        out = pd.concat([out.reset_index(drop=True), tfidf_df], axis=1)
                    except:
                        pass
                
                self.log(f"  ✅ Created features for text column: {col}")
                metrics.auto_mod()
                
            except Exception as e:
                self.log(f"  ⚠️ Text features failed for {col}: {e}")
        
        return out

    # ---- 4. Datetime Feature Engineering ------------------------------
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from datetime columns."""
        if not self.config['datetime_features']:
            return df
        
        datetime_cols = self.analysis['column_types']['datetime']
        if not datetime_cols:
            return df
        
        self.log(f"📅 Creating datetime features from {len(datetime_cols)} columns...")
        
        out = df.copy()
        
        for col in datetime_cols:
            try:
                # Ensure it's datetime
                out[col] = pd.to_datetime(out[col], errors='coerce')
                
                # Extract temporal features
                out[f"{col}_year"] = out[col].dt.year
                out[f"{col}_month"] = out[col].dt.month
                out[f"{col}_day"] = out[col].dt.day
                out[f"{col}_hour"] = out[col].dt.hour
                out[f"{col}_minute"] = out[col].dt.minute
                out[f"{col}_second"] = out[col].dt.second
                out[f"{col}_dayofweek"] = out[col].dt.dayofweek
                out[f"{col}_dayofyear"] = out[col].dt.dayofyear
                out[f"{col}_quarter"] = out[col].dt.quarter
                out[f"{col}_is_weekend"] = out[col].dt.dayofweek >= 5
                out[f"{col}_is_month_start"] = out[col].dt.is_month_start
                out[f"{col}_is_month_end"] = out[col].dt.is_month_end
                
                self.log(f"  ✅ Created features for datetime column: {col}")
                metrics.auto_mod()
                
            except Exception as e:
                self.log(f"  ⚠️ Datetime features failed for {col}: {e}")
        
        return out

    # ---- 5. Categorical Feature Engineering ---------------------------
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from categorical columns."""
        categorical_cols = self.analysis['column_types']['categorical']
        if not categorical_cols:
            return df
        
        self.log(f"🏷️ Creating categorical features from {len(categorical_cols)} columns...")
        
        out = df.copy()
        
        for col in categorical_cols:
            try:
                s = out[col].astype(str).fillna("missing")
                
                # Encoding features
                if SKLEARN_AVAILABLE:
                    le = LabelEncoder()
                    out[f"{col}_encoded"] = le.fit_transform(s)
                
                # Frequency encoding
                freq = s.value_counts(normalize=True).to_dict()
                out[f"{col}_freq"] = s.map(freq).fillna(0)
                
                # Target encoding (if target exists)
                target_col = DatasetAnalyzer.detect_target_column(df)
                if target_col and target_col in out.columns:
                    if pd.api.types.is_numeric_dtype(out[target_col]):
                        # Regression target
                        target_mean = out.groupby(col)[target_col].mean().to_dict()
                        out[f"{col}_target_mean"] = s.map(target_mean).fillna(out[target_col].mean())
                    else:
                        # Classification target - probability encoding
                        target_probs = {}
                        for val in s.unique():
                            mask = s == val
                            if mask.any():
                                target_dist = out.loc[mask, target_col].value_counts(normalize=True)
                                if not target_dist.empty:
                                    # Take the most common class probability
                                    target_probs[val] = target_dist.iloc[0]
                        out[f"{col}_target_prob"] = s.map(target_probs).fillna(0)
                
                # One-hot encoding for low cardinality
                if s.nunique() <= 10:
                    dummies = pd.get_dummies(s, prefix=col, drop_first=True)
                    out = pd.concat([out, dummies], axis=1)
                
                self.log(f"  ✅ Created features for categorical column: {col}")
                metrics.auto_mod()
                
            except Exception as e:
                self.log(f"  ⚠️ Categorical features failed for {col}: {e}")
        
        return out

    # ---- 6. Interaction Features --------------------------------------
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        if not self.config['interaction_features']:
            return df
        
        numeric_cols = self.analysis['column_types']['numeric']
        if len(numeric_cols) < 2:
            return df
        
        self.log("🔗 Creating interaction features...")
        
        out = df.copy()
        
        try:
            # Create a few key interactions
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[:2]
                out[f"{col1}_times_{col2}"] = out[col1] * out[col2]
                out[f"{col1}_div_{col2}"] = out[col1] / out[col2].replace(0, 1)
                out[f"{col1}_plus_{col2}"] = out[col1] + out[col2]
                out[f"{col1}_minus_{col2}"] = out[col1] - out[col2]
            
            # Polynomial features for top numeric columns
            if self.config['polynomial_features'] and SKLEARN_AVAILABLE:
                top_numeric = numeric_cols[:5]  # Limit to 5 columns
                if len(top_numeric) > 1:
                    poly = PolynomialFeatures(degree=self.config['max_polynomial_degree'], 
                                            include_bias=False)
                    poly_features = poly.fit_transform(out[top_numeric])
                    poly_df = pd.DataFrame(poly_features, 
                                         columns=[f"poly_{i}" for i in range(poly_features.shape[1])])
                    out = pd.concat([out.reset_index(drop=True), poly_df], axis=1)
            
            metrics.auto_mod()
            self.log("✅ Interaction features created")
            
        except Exception as e:
            self.log(f"⚠️ Interaction features failed: {e}")
        
        return out

    # ---- 7. Clustering Features ---------------------------------------
    def create_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create clustering-based features."""
        if not self.config['clustering_features'] or not SKLEARN_AVAILABLE:
            return df
        
        numeric_cols = self.analysis['column_types']['numeric']
        if len(numeric_cols) < 3:
            return df
        
        self.log("🎯 Creating clustering features...")
        
        out = df.copy()
        
        try:
            # Select top numeric columns for clustering
            cluster_cols = numeric_cols[:min(10, len(numeric_cols))]
            
            # Standardize for clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(out[cluster_cols])
            
            # K-means clustering
            for k in [2, 3, 5]:
                if k <= len(out) and k <= self.config['max_clusters']:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    out[f"cluster_k{k}"] = kmeans.fit_predict(X_scaled)
                    out[f"cluster_k{k}_dist"] = kmeans.transform(X_scaled).min(axis=1)
            
            # PCA for dimensionality reduction
            pca = PCA(n_components=min(3, len(cluster_cols)))
            pca_features = pca.fit_transform(X_scaled)
            for i in range(pca_features.shape[1]):
                out[f"pca_{i+1}"] = pca_features[:, i]
            
            metrics.auto_mod()
            self.log(f"✅ Clustering features created (explained variance: {sum(pca.explained_variance_ratio_):.2%})")
            
        except Exception as e:
            self.log(f"⚠️ Clustering features failed: {e}")
        
        return out

    # ---- 8. Feature Selection -----------------------------------------
    def select_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Select important features."""
        if not self.config['feature_selection'] or not SKLEARN_AVAILABLE:
            return df
        
        self.log("🎯 Performing feature selection...")
        
        out = df.copy()
        
        try:
            numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_col and target_col in out.columns:
                # Remove target from features
                if target_col in numeric_cols:
                    numeric_cols.remove(target_col)
                
                X = out[numeric_cols].fillna(0)
                y = out[target_col]
                
                # Choose scoring function based on target type
                if y.nunique() <= 10:  # Classification
                    selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
                else:  # Regression
                    selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
                
                X_selected = selector.fit_transform(X, y)
                selected_cols = X.columns[selector.get_support()]
                
                # Keep only selected columns plus non-numeric columns
                non_numeric_cols = [col for col in out.columns if col not in numeric_cols]
                out = pd.concat([
                    pd.DataFrame(X_selected, columns=selected_cols),
                    out[non_numeric_cols].reset_index(drop=True)
                ], axis=1)
                
                self.log(f"✅ Selected {len(selected_cols)} important features")
            
            metrics.auto_mod()
            
        except Exception as e:
            self.log(f"⚠️ Feature selection failed: {e}")
        
        return out

    # ---- 9. Main Run Function -----------------------------------------
    def run(self, data: Optional[pd.DataFrame] = None, target_column: Optional[str] = None, 
            text_columns: Optional[List[str]] = None, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Parameters:
        -----------
        data : pandas DataFrame, optional
            Input data (uses context if None)
        target_column : str, optional
            Target column name
        text_columns : list, optional
            Specific text columns to process
        config : dict, optional
            Configuration overrides
        
        Returns:
        --------
        pandas DataFrame: Engineered data
        """
        try:
            self.log("="*60)
            self.log("🎯 STARTING FEATURE ENGINEERING")
            self.log("="*60)
            
            self.status["feature_engineering"] = "running"
            
            # Resolve input data
            if data is None and self.context is not None:
                data = getattr(self.context, "transformed_data", None)
            
            if not self._is_valid_df(data):
                raise ValueError("Invalid or empty input data")
            
            df = data.copy()
            
            # Update configuration
            if config:
                self.config.update(config)
            
            # Step 1: Analyze dataset
            self.analysis = self.analyze_dataset(df)
            
            # Step 2: Create basic features
            df = self.create_basic_features(df)
            self._save_df(df, "step1_basic_features.csv")
            
            # Step 3: Text features
            df = self.create_text_features(df)
            self._save_df(df, "step2_text_features.csv")
            
            # Step 4: Datetime features
            df = self.create_datetime_features(df)
            self._save_df(df, "step3_datetime_features.csv")
            
            # Step 5: Categorical features
            df = self.create_categorical_features(df)
            self._save_df(df, "step4_categorical_features.csv")
            
            # Step 6: Interaction features
            df = self.create_interaction_features(df)
            self._save_df(df, "step5_interaction_features.csv")
            
            # Step 7: Clustering features
            df = self.create_clustering_features(df)
            self._save_df(df, "step6_clustering_features.csv")
            
            # Step 8: Feature selection
            if target_column is None:
                target_column = DatasetAnalyzer.detect_target_column(df)
            df = self.select_features(df, target_column)
            self._save_df(df, "step7_selected_features.csv")
            
            # Clean up any remaining problematic columns
            for col in df.columns:
                try:
                    # Check if column contains lists/arrays
                    if df[col].apply(lambda x: isinstance(x, (list, np.ndarray, dict))).any():
                        df = df.drop(columns=[col])
                        self.log(f"  🗑️ Dropped problematic column: {col}")
                except:
                    pass
            
            # LLM suggestions if available
            if self.llm_agent is not None:
                self.log("🤖 Consulting LLM for feature suggestions...")
                metrics.prompt_used()
                suggestion = self.llm_agent.ask(
                    f"Feature engineering completed. Dataset shape: {df.shape}. "
                    f"Target column: {target_column}. "
                    "Suggest additional features or improvements."
                )
                self.log(f"💡 LLM Suggestion: {suggestion}")
                metrics.auto_mod()
            
            # Final summary
            self.log("\n" + "="*60)
            self.log("✅ FEATURE ENGINEERING COMPLETED")
            self.log("="*60)
            self.log(f"📊 Final shape: {df.shape}")
            self.log(f"📈 Features added: {len(df.columns) - len(data.columns)}")
            self.log(f"🎯 Target column: {target_column or 'None (unsupervised)'}")
            
            if target_column and target_column in df.columns:
                if df[target_column].nunique() <= 10:
                    self.log(f"📊 Class distribution: {dict(df[target_column].value_counts())}")
            
            self.status["feature_engineering"] = "completed"
            self.engineered_data = df
            
            # Update context if available
            if self.context is not None:
                self.context.engineered_data = df
                self.context.feature_engineering_logs = self.logs
            
            return df
            
        except Exception as e:
            self.status["feature_engineering"] = "failed"
            self.log(f"❌ Feature engineering failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            metrics.correction_made()
            raise


# ============================
# Enhanced Runner
# ============================
"""
Runner: featureEngineeringRunner.py
Description: Generic feature engineering executor for any dataset.
"""

import sys
import json
from pathlib import Path


def main() -> None:
    """Main execution function."""
    # --- Command line args ---
    if len(sys.argv) < 2:
        print("Usage: python featureEngineeringRunner.py <input_file> [options]")
        print("\nOptions:")
        print("  --target <column>       : Specify target column")
        print("  --text <columns>        : Specify text columns (comma-separated)")
        print("  --config <json_file>    : Configuration file")
        print("  --output <dir>          : Output directory")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Parse arguments
    target_column = None
    text_columns = None
    config_file = None
    output_dir = "feature_outputs"
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--target" and i + 1 < len(sys.argv):
            target_column = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--text" and i + 1 < len(sys.argv):
            text_columns = sys.argv[i + 1].split(",")
            i += 2
        elif sys.argv[i] == "--config" and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # Validate input
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Load configuration if provided
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}")
    
    try:
        # --- Load data ---
        print(f"📂 Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Auto-detect columns if not provided
        if target_column is None:
            target_column = DatasetAnalyzer.detect_target_column(df)
            if target_column:
                print(f"🎯 Auto-detected target column: {target_column}")
            else:
                print("ℹ️ No target column detected")
        
        if text_columns is None:
            analyzer = DatasetAnalyzer()
            analysis = analyzer.analyze(df)
            text_columns = analysis['column_types']['text']
            if text_columns:
                print(f"📝 Auto-detected text columns: {text_columns}")
        
        # --- Initialize module ---
        module = FeatureEngineeringModule(
            context=None,
            llm_agent=None,
            save_outputs=True,
            output_dir=output_dir
        )
        
        # --- Run feature engineering ---
        print("\n" + "="*60)
        print("🔧 RUNNING FEATURE ENGINEERING")
        print("="*60)
        
        engineered_df = module.run(
            data=df,
            target_column=target_column,
            text_columns=text_columns,
            config=config
        )
        
        # --- Save results ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save engineered data
        output_path = f"{output_dir}/engineered_{timestamp}.csv"
        engineered_df.to_csv(output_path, index=False)
        print(f"\n💾 Engineered data saved to: {output_path}")
        
        # Save logs
        log_path = f"{output_dir}/logs/feature_engineering_{timestamp}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(module.logs))
        print(f"📝 Logs saved to: {log_path}")
        
        # Save analysis
        analysis_path = f"{output_dir}/analysis/analysis_{timestamp}.json"
        os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
        if module.analysis:
            with open(analysis_path, "w") as f:
                json.dump(module.analysis, f, indent=2, default=str)
            print(f"📊 Analysis saved to: {analysis_path}")
        
        print(f"\n✅ Feature Engineering completed successfully!")
        print(f"📊 Original shape: {df.shape}")
        print(f"📈 Engineered shape: {engineered_df.shape}")
        print(f"🎯 Target column: {target_column or 'None'}")
        
    except Exception as e:
        print(f"\n❌ Feature Engineering failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("FEATURE ENGINEERING MODULE - GENERIC VERSION")
    print("="*60)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, 32, 47, 51, 29],
        "income": [50000, 75000, 120000, 90000, 60000],
        "purchase_date": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10", "2024-01-25", "2024-02-28"]),
        "review": ["Great product!", "Not satisfied", "Average quality", "Excellent service", "Could be better"],
        "category": ["A", "B", "A", "C", "B"],
        "churn": [0, 1, 0, 1, 0]  # Target column
    })
    
    print("\n📊 Sample Data:")
    print(sample_data)
    
    # Test the module
    module = FeatureEngineeringModule()
    engineered = module.run(
        data=sample_data,
        target_column="churn"
    )
    
    print("\n🎯 Engineered Features Preview:")
    print(engineered.head())
    print(f"\n📈 Added {len(engineered.columns) - len(sample_data.columns)} new features")