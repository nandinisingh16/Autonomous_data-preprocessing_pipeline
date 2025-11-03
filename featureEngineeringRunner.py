"""
Runner: featureEngineeringRunner.py
Description: Executes the Feature Engineering stage of the pipeline to generate new features for modeling.
"""

import os
import sys
import pandas as pd
import datetime
from typing import Optional, List
from pipeline_context import PipelineContext
from feature_engineering import FeatureEngineeringModule

def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Detect potential text column based on data characteristics."""
    text_cols = [
        col for col in df.select_dtypes(include=["object"]).columns 
        if df[col].notna().any() and df[col].nunique() > 5
    ]
    return text_cols[0] if text_cols else None

def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Detect potential label column based on common names."""
    label_candidates = ["label", "target", "output", "class", "survived"]
    for col in df.columns:
        if col.lower() in label_candidates:
            return col
    return None

def validate_dataframe(df: Optional[pd.DataFrame], context: str) -> bool:
    """Explicit DataFrame validation."""
    if df is None:
        print(f"❌ {context}: DataFrame is None")
        return False
    if not isinstance(df, pd.DataFrame):
        print(f"❌ {context}: Not a DataFrame")
        return False
    if len(df.index) == 0:
        print(f"❌ {context}: DataFrame is empty")
        return False
    return True

def main() -> None:
    """Main execution function with proper error handling."""
    # --- Command line args ---
    if len(sys.argv) < 2:
        print("Usage: python featureEngineeringRunner.py <path_to_transformed_file> [text_column] [label_column]")
        sys.exit(1)

    transformed_file = sys.argv[1]
    text_column = sys.argv[2] if len(sys.argv) > 2 else None
    label_column = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(transformed_file):
        print(f"❌ Error: File '{transformed_file}' not found.")
        sys.exit(1)

    # --- Initialize context ---
    context = PipelineContext(stage_name="feature_engineering")

    try:
        # --- Load and validate input ---
        context.log(f"📂 Loading transformed data from: {transformed_file}")
        df = pd.read_csv(transformed_file)
        
        if not validate_dataframe(df, "Input validation"):
            raise ValueError("Invalid input DataFrame")

        # Store copy in context
        context.transformed_data = df.copy()
        context.log(f"✅ Loaded {len(df.index)} rows and {df.shape[1]} columns.")

        # --- Auto-detect columns if not provided ---
        if text_column is None:
            text_column = detect_text_column(df)
            context.log(f"🧠 Auto-detected text column: {text_column}")

        if label_column is None:
            label_column = detect_label_column(df)
            context.log(f"🎯 Auto-detected label column: {label_column}")

        # --- Run Feature Engineering ---
        fe_module = FeatureEngineeringModule(
            raw_data=df,
            llm_agent=None,
            save_outputs=True,
            output_dir="data/feature_outputs"
        )
        
        engineered_df = fe_module.run(
            text_column=text_column,
            label_column=label_column
        )

        # Validate output
        if not validate_dataframe(engineered_df, "Feature Engineering output"):
            raise ValueError("Feature Engineering produced invalid output")

        if not isinstance(engineered_df, pd.DataFrame):
            raise ValueError("Feature engineering returned no data")
            
        if len(engineered_df.index) == 0:
            raise ValueError("Feature engineering returned empty DataFrame")

        # --- Save outputs ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directories exist
        os.makedirs("data/engineered", exist_ok=True)
        os.makedirs("logs/feature_engineering", exist_ok=True)

        # Save engineered data
        output_path = f"data/engineered/engineered_{timestamp}.csv"
        engineered_df.to_csv(output_path, index=False)

        # Save logs
        log_path = f"logs/feature_engineering/feature_engineering_{timestamp}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(context.logs))

        # Update context status
        context.status["feature_engineering"] = "completed"
        context.engineered_data = engineered_df

        print("\n✅ Feature Engineering completed successfully!")
        print(f"📁 Engineered file saved to: {output_path}")
        print(f"📝 Logs saved to: {log_path}")
        print(f"ℹ️ Shape: {engineered_df.shape}")

    except Exception as e:
        context.status["feature_engineering"] = "failed"
        context.log(f"Feature Engineering failed: {str(e)}")
        print(f"\n❌ Feature Engineering stage failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
