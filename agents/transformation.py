"""
Module: transformation.py
Description: Generic data transformation module for any dataset (tabular, text, mixed).
Author: Ishita Sawhney
Date: 2025-11-02
Enhanced: 2025-12-02
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
except ImportError:
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        Tokenizer = None
        pad_sequences = None

from orchestrator.metrics_tracker import metrics


class TransformationModule:
    def __init__(self, context=None, llm_agent=None, save_outputs=True, output_dir="transformation_outputs"):
        """
        Enhanced transformation module for any dataset type.
        
        Parameters:
        -----------
        context : PipelineContext
            Pipeline context object with data and metadata
        llm_agent : LLMAgent, optional
            LLM agent for intelligent suggestions
        save_outputs : bool, default=True
            Whether to save intermediate outputs
        output_dir : str, default="transformation_outputs"
            Directory to save intermediate files
        """
        self.context = context
        self.raw_data = None
        self.llm_agent = llm_agent
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.status = {}
        self.logs = []
        self.transformed_data = None
        
        # Configuration
        self.config = {
            'scaling_method': 'minmax',  # 'minmax', 'standard', 'robust', None
            'binning_enabled': True,
            'dim_reduction_enabled': True,
            'balancing_enabled': True,
            'text_processing_enabled': True,
            'remove_low_variance': True,
            'max_text_length': 100,
            'max_categories_for_onehot': 10
        }

    def log(self, message):
        """Utility function for logging progress."""
        self.logs.append(message)
        print(f"[Transformation] {message}")

    def detect_dataset_type(self, df):
        """
        Detect the type of dataset (tabular, text, mixed, time-series).
        
        Returns:
        --------
        dict: Dataset type information
        """
        dataset_info = {
            'type': 'tabular',
            'has_text': False,
            'has_numeric': False,
            'has_categorical': False,
            'has_datetime': False,
            'has_missing': False,
            'target_present': False,
            'target_column': None,
            'is_imbalanced': False,
            'text_columns': []  # ✅ ADD THIS LINE - initialize the list
        }
        
        # Check column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        dataset_info['has_numeric'] = len(numeric_cols) > 0
        dataset_info['has_categorical'] = len(text_cols) > 0
        dataset_info['has_datetime'] = len(datetime_cols) > 0
        
        # Check for text columns (long strings)
        for col in text_cols:
            avg_length = df[col].astype(str).apply(len).mean()
            if avg_length > 20 and df[col].nunique() > len(df) * 0.5:
                dataset_info['has_text'] = True
                dataset_info['text_columns'].append(col)  # ✅ Now this will work
        
        # Check for missing values
        dataset_info['has_missing'] = df.isnull().any().any()
        
        # Detect target column
        target_col = self.detect_target_column(df)
        if target_col:
            dataset_info['target_present'] = True
            dataset_info['target_column'] = target_col
            
            # Check for imbalance
            if df[target_col].nunique() <= 10:  # Classification
                class_counts = df[target_col].value_counts()
                max_count = class_counts.max()
                min_count = class_counts.min()
                if min_count > 0 and max_count / min_count > 5:
                    dataset_info['is_imbalanced'] = True
        
        # Determine overall type
        if dataset_info['has_text'] and not dataset_info['has_numeric']:
            dataset_info['type'] = 'text'
        elif dataset_info['has_datetime'] and len(datetime_cols) > 0:
            dataset_info['type'] = 'time_series'
        elif dataset_info['has_numeric'] and dataset_info['has_categorical']:
            dataset_info['type'] = 'mixed'
        
        return dataset_info

    def detect_target_column(self, df):
        """
        Intelligently detect target column.
        
        Returns:
        --------
        str or None: Target column name
        """
        # Common target names
        common_targets = [
            'target', 'label', 'class', 'outcome', 'response',
            'survived', 'y', 'dependent', 'result', 'diagnosis',
            'churn', 'fraud', 'click', 'conversion', 'default',
            'price', 'salary', 'income', 'value', 'score'  # Regression targets
        ]
        
        # Check exact matches (case-insensitive)
        for col in df.columns:
            if col.lower() in [t.lower() for t in common_targets]:
                return col
        
        # Check for binary columns (2 unique values)
        for col in df.columns:
            unique_vals = df[col].dropna().nunique()
            if unique_vals == 2:
                return col
        
        # Check last column (common convention)
        if len(df.columns) > 0:
            last_col = df.columns[-1]
            if df[last_col].nunique() < len(df) * 0.5:  # Not too many unique values
                return last_col
        
        return None

    #############################
    # 1. MISSING VALUE HANDLING
    #############################
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values based on column type.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        strategy : str
            'auto', 'mean', 'median', 'mode', 'drop'
        """
        self.log("🔍 Handling missing values...")
        
        if strategy == 'auto':
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = missing_count / len(df)
                    
                    if missing_pct > 0.3:  # Too many missing
                        df = df.drop(columns=[col])
                        self.log(f"  Dropped column '{col}' ({missing_pct:.1%} missing)")
                    else:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing')
        
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy in ['mean', 'median', 'mode']:
            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        metrics.auto_mod()
        return df

    #############################
    # 2. SCALING / NORMALIZATION
    #############################
    def scale_normalize(self, df, method='auto'):
        """
        Scale numeric features.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        method : str
            'auto', 'minmax', 'standard', 'robust', None
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self.log("⚠️ No numeric columns to scale")
            return df
        
        self.log(f"📊 Scaling {len(numeric_cols)} numeric columns using {method}...")
        
        if method == 'auto':
            # Check distribution to choose method
            skewed_cols = []
            for col in numeric_cols:
                if df[col].skew() > 1 or df[col].skew() < -1:
                    skewed_cols.append(col)
            
            if len(skewed_cols) > len(numeric_cols) * 0.5:
                method = 'robust'
            else:
                method = 'standard'
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            return df
        
        try:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.log("✅ Scaling completed")
            metrics.auto_mod()
        except Exception as e:
            self.log(f"⚠️ Scaling failed: {e}")
        
        return df

    #############################
    # 3. ENCODING CATEGORICAL VARIABLES
    #############################
    def encode_categorical(self, df, max_categories=10):
        """
        Encode categorical variables intelligently.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        max_categories : int
            Maximum categories for one-hot encoding
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            self.log("⚠️ No categorical columns to encode")
            return df
        
        self.log(f"🏷️ Encoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            if unique_count == 2:
                # Binary encoding
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                self.log(f"  Binary encoded '{col}'")
            
            elif 3 <= unique_count <= max_categories:
                # One-hot encoding for few categories
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.log(f"  One-hot encoded '{col}' ({unique_count} categories)")
            
            else:
                # Many categories - use label encoding or frequency encoding
                if unique_count > 50:
                    # Frequency encoding for high cardinality
                    freq = df[col].value_counts(normalize=True)
                    df[col] = df[col].map(freq)
                    df[col] = df[col].fillna(0)
                    self.log(f"  Frequency encoded '{col}' ({unique_count} categories)")
                else:
                    # Label encoding
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                    self.log(f"  Label encoded '{col}' ({unique_count} categories)")
        
        metrics.auto_mod()
        return df

    #############################
    # 4. BINNING / DISCRETIZATION
    #############################
    def bin_discretize(self, df, n_bins=5, strategy='quantile'):
        """
        Bin continuous features.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        n_bins : int
            Number of bins
        strategy : str
            'quantile', 'uniform', 'kmeans'
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self.log("⚠️ No numeric columns for binning")
            return df
        
        # Only bin columns with sufficient unique values
        cols_to_bin = []
        for col in numeric_cols:
            unique_vals = df[col].nunique()
            if unique_vals > n_bins * 2:  # Enough unique values to bin
                cols_to_bin.append(col)
        
        if not cols_to_bin:
            self.log("⚠️ No suitable columns for binning")
            return df
        
        self.log(f"📦 Binning {len(cols_to_bin)} continuous columns...")
        
        try:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
            df[cols_to_bin] = discretizer.fit_transform(df[cols_to_bin])
            self.log("✅ Binning completed")
            metrics.auto_mod()
        except Exception as e:
            self.log(f"⚠️ Binning failed: {e}")
        
        return df

    #############################
    # 5. FEATURE SELECTION
    #############################
    def select_features(self, df, target_col=None, k=10):
        """
        Select most important features.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        target_col : str, optional
            Target column for supervised selection
        k : int
            Number of features to select
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) <= k:
            self.log("⚠️ Not enough features for selection")
            return df
        
        self.log(f"🎯 Selecting top {k} features...")
        
        try:
            if target_col and target_col in df.columns:
                # Supervised feature selection
                X = df.drop(columns=[target_col])
                y = df[target_col]
                selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
                X_selected = selector.fit_transform(X, y)
                selected_cols = X.columns[selector.get_support()]
                df = pd.concat([pd.DataFrame(X_selected, columns=selected_cols), 
                              y.reset_index(drop=True)], axis=1)
                self.log(f"✅ Selected {len(selected_cols)} features using supervised method")
            else:
                # Unsupervised feature selection (remove low variance)
                selector = VarianceThreshold(threshold=0.01)
                X_selected = selector.fit_transform(df[numeric_cols])
                selected_cols = np.array(numeric_cols)[selector.get_support()]
                df = pd.concat([pd.DataFrame(X_selected, columns=selected_cols),
                              df.drop(columns=numeric_cols)], axis=1)
                self.log(f"✅ Selected {len(selected_cols)} features using variance threshold")
            
            metrics.auto_mod()
        except Exception as e:
            self.log(f"⚠️ Feature selection failed: {e}")
        
        return df

    #############################
    # 6. DIMENSIONALITY REDUCTION
    #############################
    def dimensionality_reduction(self, df, target_col=None, n_components='auto'):
        """
        Reduce dimensionality using PCA or SVD.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        target_col : str, optional
            Target column
        n_components : int, float, or 'auto'
            Number of components to keep
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 5:
            self.log("⚠️ Not enough features for dimensionality reduction")
            return df
        
        self.log("🔍 Reducing dimensionality...")
        
        try:
            if n_components == 'auto':
                n_components = min(10, len(numeric_cols) // 2)
            
            # Choose method based on data sparsity
            if (df[numeric_cols] == 0).sum().sum() / (df[numeric_cols].shape[0] * df[numeric_cols].shape[1]) > 0.5:
                # Sparse data - use SVD
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
                method = 'SVD'
            else:
                # Dense data - use PCA
                reducer = PCA(n_components=n_components, random_state=42)
                method = 'PCA'
            
            reduced = reducer.fit_transform(df[numeric_cols])
            reduced_df = pd.DataFrame(reduced, 
                                    columns=[f"{method}_{i+1}" for i in range(reduced.shape[1])])
            
            # Drop original numeric columns and add reduced ones
            df = df.drop(columns=numeric_cols)
            df = pd.concat([df.reset_index(drop=True), reduced_df], axis=1)
            
            # Add target back if it was numeric
            if target_col and target_col in numeric_cols:
                df[target_col] = df[numeric_cols][target_col]
            
            explained_variance = np.sum(reducer.explained_variance_ratio_)
            self.log(f"✅ Dimensionality reduction completed ({method}, {explained_variance:.1%} variance explained)")
            metrics.auto_mod()
            
        except Exception as e:
            self.log(f"⚠️ Dimensionality reduction failed: {e}")
        
        return df

    #############################
    # 7. DATA BALANCING
    #############################
    def balance_data(self, df, target_col=None, method='auto'):
        """
        Balance imbalanced datasets.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        target_col : str, optional
            Target column name
        method : str
            'oversample', 'undersample', 'smote', 'auto'
        """
        if target_col is None:
            target_col = self.detect_target_column(df)
        
        if target_col is None:
            self.log("⚠️ No target column found for balancing")
            return df
        
        # Check if it's a classification problem
        unique_classes = df[target_col].nunique()
        if unique_classes > 20 or unique_classes < 2:
            self.log("⚠️ Not a classification problem or too many classes")
            return df
        
        self.log(f"⚖️ Balancing data for target '{target_col}'...")
        
        try:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            if method == 'auto':
                # Choose method based on imbalance ratio
                class_counts = Counter(y)
                majority_class = max(class_counts, key=class_counts.get)
                minority_class = min(class_counts, key=class_counts.get)
                imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
                
                if imbalance_ratio > 10:
                    method = 'undersample'  # Severe imbalance
                elif imbalance_ratio > 3:
                    method = 'smote'  # Moderate imbalance
                else:
                    method = 'oversample'  # Mild imbalance
            
            if method == 'oversample':
                balancer = RandomOverSampler(random_state=42)
            elif method == 'undersample':
                balancer = RandomUnderSampler(random_state=42)
            elif method == 'smote':
                balancer = SMOTE(random_state=42)
            else:
                self.log(f"⚠️ Unknown balancing method: {method}")
                return df
            
            X_resampled, y_resampled = balancer.fit_resample(X, y)
            df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                                   pd.Series(y_resampled, name=target_col)], axis=1)
            
            self.log(f"✅ Data balanced using {method}")
            self.log(f"  Before: {dict(Counter(y))}")
            self.log(f"  After: {dict(Counter(y_resampled))}")
            metrics.auto_mod()
            
            return df_balanced
            
        except Exception as e:
            self.log(f"⚠️ Data balancing failed: {e}")
            return df

    #############################
    # 8. TEXT PROCESSING
    #############################
    def process_text_features(self, df, text_columns=None):
        """
        Process text columns.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        text_columns : list, optional
            List of text columns to process
        """
        if text_columns is None:
            # Auto-detect text columns (long strings)
            text_columns = []
            for col in df.select_dtypes(include=['object']).columns:
                avg_length = df[col].astype(str).apply(len).mean()
                if avg_length > 20 and df[col].nunique() > len(df) * 0.1:
                    text_columns.append(col)
        
        if not text_columns:
            self.log("⚠️ No text columns to process")
            return df
        
        self.log(f"📝 Processing {len(text_columns)} text columns...")
        
        try:
            for col in text_columns:
                # Basic text features
                df[f'{col}_length'] = df[col].astype(str).apply(len)
                df[f'{col}_word_count'] = df[col].astype(str).apply(lambda x: len(x.split()))
                df[f'{col}_has_uppercase'] = df[col].astype(str).apply(lambda x: any(c.isupper() for c in x))
                df[f'{col}_has_digits'] = df[col].astype(str).apply(lambda x: any(c.isdigit() for c in x))
                
                # Tokenization if keras is available
                if Tokenizer is not None:
                    try:
                        tokenizer = Tokenizer(num_words=1000)
                        tokenizer.fit_on_texts(df[col].astype(str))
                        sequences = tokenizer.texts_to_sequences(df[col].astype(str))
                        
                        if pad_sequences is not None:
                            padded = pad_sequences(sequences, maxlen=self.config['max_text_length'])
                            # ⚠️ FIX: Don't store lists in DataFrame - it causes unhashable type errors
                            # Instead, store as string representation or skip
                            df[f'{col}_sequence_length'] = [len(seq) for seq in padded]
                            df[f'{col}_sequence_sum'] = [sum(seq) for seq in padded]
                            self.log(f"  Created sequence features for '{col}' (avoiding list storage)")
                    except Exception as e:
                        self.log(f"  ⚠️ Tokenization failed for '{col}': {e}")
                
                self.log(f"  Processed text column '{col}'")
            
            metrics.auto_mod()
            
        except Exception as e:
            self.log(f"⚠️ Text processing failed: {e}")
        
        return df
    #############################
    # 9. DATETIME PROCESSING
    #############################
    def process_datetime_features(self, df):
        """
        Extract features from datetime columns.
        """
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not datetime_cols:
            return df
        
        self.log(f"📅 Processing {len(datetime_cols)} datetime columns...")
        
        for col in datetime_cols:
            try:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
                df[f'{col}_quarter'] = df[col].dt.quarter
                
                self.log(f"  Extracted features from datetime column '{col}'")
            except:
                self.log(f"  Could not process datetime column '{col}'")
        
        return df

    #############################
    # 10. SAVE INTERMEDIATE RESULTS
    #############################
    def save_intermediate(self, df, step_name):
        """Save intermediate processed DataFrame."""
        if self.save_outputs:
            filename = f"step_{len(os.listdir(self.output_dir))+1:02d}_{step_name}.csv"
            output_path = os.path.join(self.output_dir, filename)
            df.to_csv(output_path, index=False)
            self.log(f"💾 Saved: {output_path}")
            return output_path
        return None

    #############################
    # 11. MAIN RUN FUNCTION
    #############################
    def run(self, data=None, target_column=None, text_columns=None, config=None):
        """
        Main transformation pipeline.
        
        Parameters:
        -----------
        data : pandas DataFrame, optional
            Input data (if None, uses context)
        target_column : str, optional
            Target column name
        text_columns : list, optional
            List of text columns
        config : dict, optional
            Configuration overrides
        
        Returns:
        --------
        pandas DataFrame: Transformed data
        """
        try:
            self.log("🔄 Starting transformation pipeline...")
            
            # Resolve input data
            if data is None and self.context is not None:
                data = getattr(self.context, "cleaned_data", None) or getattr(self.context, "ingested_data", None)
            
            if data is None:
                raise ValueError("No input data provided")
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(data)}")
            
            if data.empty:
                self.log("⚠️ Input DataFrame is empty")
                return data
            
            df = data.copy()
            
            # Update configuration
            if config:
                self.config.update(config)
            
            # Detect dataset characteristics
            dataset_info = self.detect_dataset_type(df)
            self.log(f"📊 Dataset type: {dataset_info['type']}")
            self.log(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Use detected target if not provided
            if target_column is None and dataset_info['target_present']:
                target_column = dataset_info['target_column']
                if target_column:
                    self.log(f"🎯 Auto-detected target column: '{target_column}'")
            
            # If text_columns not provided, use auto-detected ones
            if text_columns is None and dataset_info['text_columns']:
                text_columns = dataset_info['text_columns']
                if text_columns:
                    self.log(f"📝 Using auto-detected text columns: {text_columns}")
            
            # Step 1: Handle missing values
            df = self.handle_missing_values(df, strategy='auto')
            self.save_intermediate(df, "missing_handled.csv")
            
            # Step 2: Process datetime features
            df = self.process_datetime_features(df)
            
            # Step 3: Process text features
            if self.config['text_processing_enabled'] and text_columns:
                df = self.process_text_features(df, text_columns)
            elif text_columns:
                self.log(f"ℹ️ Text processing disabled for columns: {text_columns}")
            
            # Step 4: Encode categorical variables
            df = self.encode_categorical(df, max_categories=self.config['max_categories_for_onehot'])
            self.save_intermediate(df, "encoded.csv")
            
            # Step 5: Scale numeric features
            if self.config['scaling_method']:
                df = self.scale_normalize(df, method=self.config['scaling_method'])
                self.save_intermediate(df, "scaled.csv")
            
            # Step 6: Feature selection
            df = self.select_features(df, target_column, k=20)
            
            # Step 7: Dimensionality reduction
            if self.config['dim_reduction_enabled'] and len(df.columns) > 15:
                df = self.dimensionality_reduction(df, target_column, n_components='auto')
                self.save_intermediate(df, "reduced.csv")
            
            # Step 8: Binning
            if self.config['binning_enabled']:
                df = self.bin_discretize(df, n_bins=5, strategy='quantile')
                self.save_intermediate(df, "binned.csv")
            
            # Step 9: Data balancing
            if self.config['balancing_enabled'] and target_column and dataset_info['is_imbalanced']:
                df = self.balance_data(df, target_column, method='auto')
                self.save_intermediate(df, "balanced.csv")
            
            # 🔧 CLEANUP: Remove any columns that contain lists/arrays
            columns_to_drop = []
            for col in df.columns:
                try:
                    # Check if column contains any lists or arrays
                    sample = df[col].dropna().head(10) if not df[col].empty else pd.Series([])
                    if not sample.empty:
                        # Check first few non-null values
                        for val in sample:
                            if isinstance(val, (list, np.ndarray, dict)):
                                columns_to_drop.append(col)
                                break
                except:
                    # If we can't check, drop the column to be safe
                    columns_to_drop.append(col)
            
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                self.log(f"🧹 Cleaned up {len(columns_to_drop)} problematic columns: {columns_to_drop}")
            
            # LLM suggestion if available
            if self.llm_agent is not None:
                metrics.prompt_used()
                suggestion = self.llm_agent.ask(
                    f"Transformation completed. Dataset shape: {df.shape}. "
                    f"Target column: {target_column}. "
                    "Suggest any additional transformations or improvements."
                )
                self.log(f"💡 LLM Suggestion: {suggestion}")
                metrics.auto_mod()
            
            # Final summary
            self.log("\n" + "="*60)
            self.log("✅ TRANSFORMATION COMPLETED SUCCESSFULLY")
            self.log("="*60)
            self.log(f"📊 Final shape: {df.shape}")
            self.log(f"🎯 Target column: {target_column or 'None'}")
            self.log(f"💾 Output saved to: {self.output_dir}")
            
            if target_column and target_column in df.columns:
                if df[target_column].nunique() <= 10:
                    self.log(f"📈 Class distribution: {dict(df[target_column].value_counts())}")
                else:
                    self.log(f"📈 Target statistics: mean={df[target_column].mean():.2f}, "
                        f"std={df[target_column].std():.2f}")
            
            self.status["transformation"] = "completed"
            self.transformed_data = df
            
            # Update context if available
            if self.context is not None:
                self.context.transformed_data = df
                self.context.transformation_logs = self.logs
            
            return df
            
        except Exception as e:
            self.status["transformation"] = "failed"
            self.log(f"❌ Transformation failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            metrics.correction_made()
            return False
    # ============================
# EXAMPLE USAGE
# ============================
if __name__ == "__main__":
    # Example 1: Generic tabular data (not hardcoded to Titanic)
    print("="*60)
    print("EXAMPLE 1: GENERIC TABULAR DATA")
    print("="*60)

    # Create a generic sample dataset instead of Titanic-specific
    sample_df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],  # Generic target name
        "category": ["A", "B", "A", "C", "B"],
        "description": ["Short text", "This is a longer description with more words",
                       "Another description", "Yet another text field", "Final description"],
        "numeric": [10.5, 20.3, 15.7, 8.9, 12.1],
        "binary": [True, False, True, False, True]
    })

    transformer = TransformationModule()
    transformed = transformer.run(
        data=sample_df,
        target_column=None,  # Let auto-detection work
        text_columns=None,   # Let auto-detection work
        config={'scaling_method': 'minmax', 'balancing_enabled': False}
    )

    print("\nTransformed Data Preview:")
    print(transformed.head())

    # Example 2: Text data
    print("\n" + "="*60)
    print("EXAMPLE 2: TEXT DATA")
    print("="*60)

    text_df = pd.DataFrame({
        "text": [
            "This is a positive review about the product",
            "Negative experience with customer service",
            "The item works as expected",
            "Very disappointed with the quality",
            "Excellent product, highly recommended"
        ],
        "sentiment": ["positive", "negative", "neutral", "negative", "positive"]
    })

    transformer2 = TransformationModule()
    transformed2 = transformer2.run(
        data=text_df,
        target_column=None,  # Auto-detect target
        config={'text_processing_enabled': True}
    )

    print("\nText Data Transformed:")
    print(transformed2.head())
