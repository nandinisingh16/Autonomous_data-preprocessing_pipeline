"""
Runner: featureEngineeringRunner.py
Description: Generic feature engineering executor for any dataset type.
Enhanced: 2025-12-02
"""

import os
import sys
import pandas as pd
import numpy as np
import datetime
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import required modules with fallbacks
try:
    from pipeline_context import PipelineContext
except ImportError:
    class PipelineContext:
        def __init__(self, stage_name="feature_engineering"):
            self.stage_name = stage_name
            self.logs = []
            self.status = {}
            self.data = {}
        
        def log(self, message):
            self.logs.append(message)
            print(f"[{self.stage_name}] {message}")

try:
    from feature_engineering import FeatureEngineeringModule
except ImportError:
    print("⚠️ FeatureEngineeringModule not found")
    FeatureEngineeringModule = None

from metrics_tracker import metrics


class DatasetDetector:
    """Intelligent dataset column detection and analysis."""
    
    @staticmethod
    def detect_text_columns(df: pd.DataFrame) -> List[str]:
        """Detect text columns based on characteristics."""
        text_cols = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            if df[col].notna().any():
                # Check if it's likely text (not just categorical)
                unique_ratio = df[col].nunique() / len(df)
                avg_length = df[col].astype(str).apply(len).mean()
                
                # Heuristics for text detection
                is_text = (
                    (unique_ratio > 0.1) or  # Many unique values
                    (avg_length > 20) or      # Long strings
                    ("text" in col.lower()) or  # Column name hint
                    ("desc" in col.lower()) or
                    ("review" in col.lower()) or
                    ("comment" in col.lower())
                )
                
                if is_text:
                    text_cols.append(col)
        
        return text_cols
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        """Intelligently detect target column."""
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
        
        # Check for binary columns (2 unique values)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    return col
        
        # Check last column (common convention)
        if len(df.columns) > 0:
            last_col = df.columns[-1]
            unique_vals = df[last_col].nunique()
            # Not too many unique values (likely not an ID)
            if unique_vals < len(df) * 0.5:
                return last_col
        
        # Check for columns with low cardinality
        for col in df.columns:
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 10:  # Likely classification target
                return col
        
        return None
    
    @staticmethod
    def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
        """Detect datetime columns."""
        datetime_cols = []
        
        # Check dtype first
        for col in df.select_dtypes(include=['datetime64']).columns:
            datetime_cols.append(col)
        
        # Try to convert string columns that look like dates
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Sample some values to check
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    # Try to parse as datetime
                    pd.to_datetime(sample, errors='raise')
                    datetime_cols.append(col)
            except:
                pass
        
        return datetime_cols
    
    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset analysis."""
        analysis = {
            'shape': df.shape,
            'column_types': {
                'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'text': [],
                'datetime': [],
                'binary': [],
                'id_like': []
            },
            'missing_values': {},
            'target_candidates': []
        }
        
        # Classify columns more precisely
        for col in df.columns:
            # Missing values
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            analysis['missing_values'][col] = {
                'count': int(missing),
                'percentage': float(missing_pct)
            }
            
            # Check if binary
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                analysis['column_types']['binary'].append(col)
            
            # Check if ID-like
            col_lower = col.lower()
            if ('id' in col_lower or 'index' in col_lower or 
                'key' in col_lower or unique_vals == len(df)):
                analysis['column_types']['id_like'].append(col)
        
        # Detect text columns
        analysis['column_types']['text'] = DatasetDetector.detect_text_columns(df)
        
        # Detect datetime columns
        analysis['column_types']['datetime'] = DatasetDetector.detect_datetime_columns(df)
        
        # Remove overlaps
        for col_type in ['text', 'datetime', 'binary', 'id_like']:
            for col in analysis['column_types'][col_type]:
                if col in analysis['column_types']['categorical']:
                    analysis['column_types']['categorical'].remove(col)
        
        # Find target candidates
        target_col = DatasetDetector.detect_target_column(df)
        if target_col:
            analysis['target_candidates'].append({
                'column': target_col,
                'type': 'auto_detected',
                'unique_values': int(df[target_col].nunique())
            })
        
        return analysis


def validate_dataframe(df: Optional[pd.DataFrame], context: str = "DataFrame") -> bool:
    """Validate DataFrame with detailed error messages."""
    if df is None:
        print(f"❌ {context}: DataFrame is None")
        return False
    
    if not isinstance(df, pd.DataFrame):
        print(f"❌ {context}: Expected DataFrame, got {type(df)}")
        return False
    
    if df.empty:
        print(f"❌ {context}: DataFrame is empty (0 rows)")
        return False
    
    if len(df.columns) == 0:
        print(f"❌ {context}: DataFrame has no columns")
        return False
    
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Runner for any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python featureEngineeringRunner.py data.csv
  python featureEngineeringRunner.py data.csv --target churn
  python featureEngineeringRunner.py data.csv --text review description
  python featureEngineeringRunner.py data.csv --config config.json
  python featureEngineeringRunner.py data.csv --output ./results
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--target", "-t",
        dest="target_column",
        help="Target column name"
    )
    
    parser.add_argument(
        "--text", "-x",
        nargs="+",
        dest="text_columns",
        help="Text columns to process"
    )
    
    parser.add_argument(
        "--config", "-c",
        dest="config_file",
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--output", "-o",
        dest="output_dir",
        default="feature_engineering_outputs",
        help="Output directory (default: feature_engineering_outputs)"
    )
    
    parser.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        default=True,
        help="Disable LLM suggestions"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_configuration(config_file: Optional[str]) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    default_config = {
        'text_features': True,
        'datetime_features': True,
        'interaction_features': True,
        'clustering_features': True,
        'feature_selection': True,
        'max_polynomial_degree': 2,
        'max_text_features': 10,
        'max_clusters': 5,
        'save_intermediate': True
    }
    
    if not config_file or not os.path.exists(config_file):
        return default_config
    
    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        config = {**default_config, **user_config}
        print(f"✅ Loaded configuration from {config_file}")
        return config
    except Exception as e:
        print(f"⚠️ Failed to load config file: {e}")
        return default_config


def main() -> None:
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    input_file = args.input_file
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Load configuration
    config = load_configuration(args.config_file)
    
    # Initialize context
    context = PipelineContext(stage_name="feature_engineering")
    
    try:
        print("="*60)
        print("🎯 FEATURE ENGINEERING RUNNER")
        print("="*60)
        
        # --- Load and validate input ---
        context.log(f"📂 Loading data from: {input_file}")
        
        # Try different encodings if needed
        try:
            df = pd.read_csv(input_file)
        except UnicodeDecodeError:
            context.log("⚠️ UTF-8 failed, trying latin-1 encoding...")
            df = pd.read_csv(input_file, encoding='latin-1')
        except Exception as e:
            context.log(f"❌ Failed to read CSV: {e}")
            sys.exit(1)
        
        if not validate_dataframe(df, "Input validation"):
            raise ValueError("Invalid input DataFrame")
        
        context.log(f"✅ Loaded {len(df)} rows and {df.shape[1]} columns.")
        
        # Store in context
        context.transformed_data = df.copy()
        
        # --- Analyze dataset ---
        context.log("\n🔍 Analyzing dataset...")
        detector = DatasetDetector()
        analysis = detector.analyze_dataset(df)
        
        context.log(f"📊 Dataset shape: {analysis['shape']}")
        context.log(f"📈 Column analysis:")
        for col_type, cols in analysis['column_types'].items():
            if cols:
                context.log(f"  - {col_type}: {len(cols)} columns")
                if args.verbose and cols:
                    context.log(f"    {cols}")
        
        # Check missing values
        high_missing = []
        for col, stats in analysis['missing_values'].items():
            if stats['percentage'] > 30:
                high_missing.append((col, stats['percentage']))
        
        if high_missing:
            context.log("⚠️  High missing values (>30%):")
            for col, pct in high_missing[:5]:  # Show top 5
                context.log(f"  - {col}: {pct:.1f}%")
        
        # --- Auto-detect columns if not provided ---
        target_column = args.target_column
        if target_column is None:
            target_column = detector.detect_target_column(df)
            if target_column:
                context.log(f"🎯 Auto-detected target column: '{target_column}'")
            else:
                context.log("ℹ️ No target column detected (unsupervised)")
        
        text_columns = args.text_columns
        if text_columns is None:
            text_columns = detector.detect_text_columns(df)
            if text_columns:
                context.log(f"📝 Auto-detected text columns: {text_columns}")
            else:
                context.log("ℹ️ No text columns detected")
        
        datetime_columns = detector.detect_datetime_columns(df)
        if datetime_columns:
            context.log(f"📅 Detected datetime columns: {datetime_columns}")
        
        # --- Initialize Feature Engineering Module ---
        context.log("\n🔧 Initializing Feature Engineering Module...")
        
        # Create LLM agent if needed
        llm_agent = None
        if args.use_llm:
            try:
                # Try to import LLM agent
                from llm_agent import create_llm_agent
                llm_agent = create_llm_agent(use_llm=True, provider="openai")
                context.log("🤖 LLM agent initialized")
            except ImportError:
                context.log("⚠️ LLM agent not available")
        
        fe_module = FeatureEngineeringModule(
            context=context,
            llm_agent=llm_agent,
            save_outputs=config.get('save_intermediate', True),
            output_dir=args.output_dir
        )
        
        # Update module config
        fe_module.config.update(config)
        
        # --- Run Feature Engineering ---
        context.log("\n" + "="*60)
        context.log("🚀 RUNNING FEATURE ENGINEERING")
        context.log("="*60)
        
        engineered_df = fe_module.run(
            data=df,
            target_column=target_column,
            text_columns=text_columns,
            config=config
        )
        
        # Validate output
        if not validate_dataframe(engineered_df, "Feature Engineering output"):
            raise ValueError("Feature Engineering produced invalid output")
        
        # --- Save outputs ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        output_dirs = {
            'engineered': os.path.join(args.output_dir, "engineered"),
            'logs': os.path.join(args.output_dir, "logs"),
            'analysis': os.path.join(args.output_dir, "analysis"),
            'reports': os.path.join(args.output_dir, "reports")
        }
        
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Save engineered data
        output_path = os.path.join(output_dirs['engineered'], f"engineered_{timestamp}.csv")
        engineered_df.to_csv(output_path, index=False)
        
        # Save logs
        log_path = os.path.join(output_dirs['logs'], f"feature_engineering_{timestamp}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(context.logs))
        
        # Save analysis
        analysis_path = os.path.join(output_dirs['analysis'], f"analysis_{timestamp}.json")
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save summary report
        summary_path = os.path.join(output_dirs['reports'], f"summary_{timestamp}.txt")
        with open(summary_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("FEATURE ENGINEERING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Input File: {input_file}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Original Shape: {df.shape}\n")
            f.write(f"Engineered Shape: {engineered_df.shape}\n")
            f.write(f"Features Added: {len(engineered_df.columns) - len(df.columns)}\n\n")
            f.write(f"Target Column: {target_column or 'None'}\n")
            if target_column and target_column in engineered_df.columns:
                unique_vals = engineered_df[target_column].nunique()
                f.write(f"Target Unique Values: {unique_vals}\n")
                if unique_vals <= 10:
                    value_counts = engineered_df[target_column].value_counts()
                    f.write(f"Target Distribution:\n")
                    for val, count in value_counts.items():
                        f.write(f"  {val}: {count} ({count/len(engineered_df):.1%})\n")
            f.write(f"\nText Columns Processed: {text_columns or 'None'}\n")
            f.write(f"\nOutput Files:\n")
            f.write(f"  Engineered Data: {output_path}\n")
            f.write(f"  Logs: {log_path}\n")
            f.write(f"  Analysis: {analysis_path}\n")
            f.write(f"  Summary: {summary_path}\n")
        
        # Update context
        context.status["feature_engineering"] = "completed"
        context.engineered_data = engineered_df
        
        # --- Final Output ---
        print("\n" + "="*60)
        print("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📊 Results Summary:")
        print(f"  Original shape: {df.shape}")
        print(f"  Engineered shape: {engineered_df.shape}")
        print(f"  Features added: {len(engineered_df.columns) - len(df.columns)}")
        print(f"  Target column: {target_column or 'None (unsupervised)'}")
        
        if target_column and target_column in engineered_df.columns:
            unique_vals = engineered_df[target_column].nunique()
            print(f"  Target unique values: {unique_vals}")
            if unique_vals <= 10:
                print(f"  Target distribution:")
                for val, count in engineered_df[target_column].value_counts().items():
                    print(f"    {val}: {count} ({count/len(engineered_df):.1%})")
        
        print(f"\n💾 Output Files:")
        print(f"  Engineered data: {output_path}")
        print(f"  Logs: {log_path}")
        print(f"  Analysis: {analysis_path}")
        print(f"  Summary: {summary_path}")
        
        # Show sample of new features
        new_columns = [col for col in engineered_df.columns if col not in df.columns]
        if new_columns:
            print(f"\n✨ New Features Created ({len(new_columns)} total):")
            # Group by feature type
            feature_types = {}
            for col in new_columns:
                if '_text_' in col or any(txt in col for txt in text_columns or []):
                    feature_types.setdefault('Text Features', []).append(col)
                elif '_datetime_' in col or any(dt in col for dt in datetime_columns or []):
                    feature_types.setdefault('Datetime Features', []).append(col)
                elif '_cluster_' in col or col.startswith('pca_') or col.startswith('poly_'):
                    feature_types.setdefault('Clustering/Reduction', []).append(col)
                elif '_encoded' in col or '_freq' in col or '_target_' in col:
                    feature_types.setdefault('Categorical Encoding', []).append(col)
                elif 'numeric_' in col or 'missing_' in col:
                    feature_types.setdefault('Statistical Features', []).append(col)
                else:
                    feature_types.setdefault('Other Features', []).append(col)
            
            for feature_type, features in feature_types.items():
                print(f"  {feature_type} ({len(features)}):")
                for feat in features[:5]:  # Show first 5 of each type
                    print(f"    - {feat}")
                if len(features) > 5:
                    print(f"    ... and {len(features) - 5} more")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Review the engineered features in: {output_path}")
        print(f"  2. Check the analysis report: {analysis_path}")
        print(f"  3. Proceed to modeling with the engineered dataset")
        
    except Exception as e:
        context.status["feature_engineering"] = "failed"
        error_msg = f"Feature Engineering failed: {str(e)}"
        context.log(error_msg)
        print(f"\n❌ {error_msg}")
        
        # Save error log
        if 'output_dir' in locals():
            error_log_path = os.path.join(args.output_dir, "error.log")
            with open(error_log_path, "w") as f:
                f.write(f"Feature Engineering Error: {str(e)}\n\n")
                import traceback
                f.write(traceback.format_exc())
            print(f"📝 Error details saved to: {error_log_path}")
        
        sys.exit(1)


if __name__ == "__main__":
    # Example usage if run without arguments
    if len(sys.argv) == 1:
        print("="*60)
        print("FEATURE ENGINEERING RUNNER - GENERIC VERSION")
        print("="*60)
        print("\nUsage examples:")
        print("  python featureEngineeringRunner.py data.csv")
        print("  python featureEngineeringRunner.py data.csv --target churn")
        print("  python featureEngineeringRunner.py data.csv --text review description")
        print("  python featureEngineeringRunner.py data.csv --config config.json")
        print("\nUse --help for all options")
        sys.exit(0)
    
    main()