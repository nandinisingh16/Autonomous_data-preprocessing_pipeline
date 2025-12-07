"""
File: visualize_pipeline_results.py
Purpose: Generic visualization for results from the Autonomous Data Preprocessing Pipeline
Author: Raj Nandini Singh
Date: 2025-11-03
Enhanced: 2025-12-02
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# === CONFIG ===
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
sns.set(font_scale=1.1)
plt.rcParams['figure.autolayout'] = True

def detect_target_column(df, user_target=None):
    """
    Intelligently detect target column in a dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
    user_target : str, optional
        User-specified target column
    
    Returns:
    --------
    str or None: Detected target column name
    """
    
    # If user specified target, use it
    if user_target and user_target in df.columns:
        return user_target
    
    # Common target column names (case-insensitive)
    common_targets = [
        'target', 'label', 'class', 'outcome', 'response', 
        'survived', 'y', 'dependent', 'result', 'diagnosis',
        'churn', 'fraud', 'click', 'conversion', 'default'
    ]
    
    # Check exact matches first
    for col in df.columns:
        if col.lower() in [t.lower() for t in common_targets]:
            return col
    
    # Check for binary/categorical columns
    for col in df.columns:
        unique_vals = df[col].dropna().nunique()
        
        # Binary classification (exactly 2 unique values)
        if unique_vals == 2:
            print(f"🔍 Auto-detected binary target: '{col}'")
            return col
        
        # Multiclass classification (3-20 unique values, not too many)
        elif 3 <= unique_vals <= 20:
            # Check if it's not a categorical ID column
            if df[col].dtype in ['object', 'category'] or unique_vals < len(df) * 0.1:
                print(f"🔍 Auto-detected categorical target: '{col}' ({unique_vals} classes)")
                return col
    
    # Check last column (common convention)
    if len(df.columns) > 0:
        last_col = df.columns[-1]
        unique_vals = df[last_col].dropna().nunique()
        if unique_vals <= 20:
            print(f"🔍 Using last column as potential target: '{last_col}'")
            return last_col
    
    return None

def analyze_feature_importance(df, target_col, top_n=10):
    """
    Analyze feature importance for classification or regression
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
    target_col : str
        Target column name
    top_n : int, default=10
        Number of top features to display
    """
    
    print(f"\n🎯 Feature Importance Analysis for target: '{target_col}'")
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle missing values
    y_clean = y.dropna()
    X_clean = X.loc[y_clean.index]
    
    # Select numeric features
    X_numeric = X_clean.select_dtypes(include=[np.number])
    
    # Handle non-numeric features by encoding
    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        X_encoded = X_numeric.copy()
        for col in non_numeric_cols:
            try:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_clean[col].astype(str).fillna('missing'))
            except:
                print(f"  ⚠ Skipping non-numeric column '{col}' for importance analysis")
        X_numeric = X_encoded
    
    # Fill NaN with median
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    if len(X_numeric.columns) == 0:
        print("  ❌ No suitable features for importance analysis")
        return
    
    # Determine problem type
    unique_classes = y_clean.nunique()
    
    if unique_classes == 2:
        # Binary classification
        print(f"  📊 Binary classification detected ({y_clean.unique()})")
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        y_encoded = y_clean
    elif 3 <= unique_classes <= 20:
        # Multiclass classification
        print(f"  📊 Multiclass classification detected ({unique_classes} classes)")
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        y_encoded = LabelEncoder().fit_transform(y_clean)
    else:
        # Regression
        print(f"  📊 Regression detected ({unique_classes} unique values)")
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        y_encoded = y_clean
    
    # Fit model
    try:
        model.fit(X_numeric, y_encoded)
        
        # Get feature importance
        importance = pd.Series(model.feature_importances_, index=X_numeric.columns)
        importance = importance.sort_values(ascending=False)
        
        # Display top features
        print(f"\n🏆 Top {min(top_n, len(importance))} Most Important Features:")
        for i, (feat, imp) in enumerate(importance.head(top_n).items(), 1):
            print(f"  {i:2d}. {feat:<30} {imp:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        top_features = importance.head(top_n)
        
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
        plt.title(f"Top {len(top_features)} Feature Importances for '{target_col}'", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Also show cumulative importance
        plt.figure(figsize=(10, 5))
        cumulative_importance = importance.cumsum()
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance.values, 
                marker='o', linewidth=2, markersize=6)
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% Threshold')
        plt.xlabel("Number of Features", fontsize=12)
        plt.ylabel("Cumulative Importance", fontsize=12)
        plt.title("Cumulative Feature Importance", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Find number of features needed for 80% importance
        features_for_80 = (cumulative_importance <= 0.8).sum() + 1
        print(f"  📈 {features_for_80} features capture 80% of importance")
        
    except Exception as e:
        print(f"  ❌ Error in feature importance analysis: {str(e)}")

# === MAIN VISUALIZATION FUNCTION ===
def visualize_pipeline_results(file_path, target_col=None):
    """
    Main visualization function for pipeline results
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    target_col : str, optional
        Specific target column name
    """
    
    # === LOAD FINAL DATA ===
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ {file_path} not found.")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {file_path} with shape {df.shape}")
    print(f"📁 File size: {os.path.getsize(file_path) / 1024:.1f} KB")
    
    # === QUICK OVERVIEW ===
    print("\n" + "="*60)
    print("🔍 DATA OVERVIEW")
    print("="*60)
    
    print(f"\n📋 Column Types:")
    print(df.dtypes.value_counts())
    
    print(f"\n📊 Dataset Info:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n❌ Missing Values:")
    missing_summary = df.isnull().sum()
    missing_total = missing_summary.sum()
    missing_percent = (missing_total / (len(df) * len(df.columns))) * 100
    print(f"  Total missing: {missing_total:,} ({missing_percent:.1f}%)")
    if missing_total > 0:
        top_missing = missing_summary[missing_summary > 0].sort_values(ascending=False).head(5)
        for col, count in top_missing.items():
            percent = (count / len(df)) * 100
            print(f"    {col}: {count:,} ({percent:.1f}%)")
    
    # === 1. Missing Value Heatmap ===
    if missing_total > 0:
        print("\n📊 Generating Missing Value Heatmap...")
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='Reds', yticklabels=False)
        plt.title("Missing Value Heatmap", fontsize=14, fontweight='bold')
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Rows (sampled)", fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("\n✅ No missing values found in the dataset")
    
    # === 2. Correlation Heatmap (Numerical) ===
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        print(f"\n📈 Analyzing {len(num_cols)} numerical features...")
        
        # Filter columns with variance
        num_cols_with_variance = [col for col in num_cols if df[col].std() > 0]
        
        if len(num_cols_with_variance) > 1:
            plt.figure(figsize=(max(10, len(num_cols_with_variance)//2), 
                              max(8, len(num_cols_with_variance)//2)))
            
            corr_matrix = df[num_cols_with_variance].corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .8})
            plt.title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Find highly correlated features
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            if corr_pairs:
                print("\n⚠️  Highly Correlated Features (|r| > 0.8):")
                for feat1, feat2, corr_val in corr_pairs[:10]:  # Show top 10
                    print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            print("⚠️ Not enough numerical features with variance for correlation analysis")
    
    # === 3. Distribution of Numeric Features ===
    if len(num_cols) > 0:
        print(f"\n📊 Plotting distributions for {len(num_cols)} numerical features...")
        
        # Plot in a grid
        n_cols = 3
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(num_cols):
            if idx < len(axes):
                axes[idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[idx].set_title(col, fontsize=10)
                axes[idx].set_xlabel('')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(num_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # === 4. Boxplots for Top Numerical Features (Outlier Detection) ===
    if len(num_cols) > 0:
        print("\n🔍 Checking for outliers...")
        # Show boxplots for top 9 features
        top_features = num_cols[:9]
        
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(top_features):
            if idx < len(axes):
                sns.boxplot(y=df[col], ax=axes[idx], color='skyblue')
                axes[idx].set_title(f"{col}", fontsize=10)
                axes[idx].set_xlabel('')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Outlier Detection - Boxplots", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # === 5. Count Plots for Categorical Columns ===
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(cat_cols) > 0:
        print(f"\n📊 Analyzing {len(cat_cols)} categorical features...")
        
        for col in cat_cols:
            unique_count = df[col].nunique()
            
            if unique_count <= 15:  # Reasonable for visualization
                plt.figure(figsize=(8, 4))
                value_counts = df[col].value_counts()
                
                if unique_count > 5:
                    # For many categories, show horizontal bar chart
                    value_counts.sort_values().plot(kind='barh')
                    plt.ylabel(col)
                    plt.xlabel('Count')
                else:
                    # For few categories, show pie chart
                    plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                    plt.ylabel('')
                
                plt.title(f"Distribution of '{col}' ({unique_count} unique values)", fontsize=12)
                plt.tight_layout()
                plt.show()
                
                print(f"  {col}: {unique_count} unique values")
            else:
                print(f"  ⚠ {col}: Skipped (too many unique values: {unique_count})")
    
    # === 6. Pairplot for Top Numerical Features ===
    if len(num_cols) > 2:
        print("\n📈 Generating pairplot for relationships...")
        sample_cols = num_cols[:6]  # Limit to 6 for readability
        
        # Try to detect target for coloring
        target_col_detected = detect_target_column(df, target_col)
        
        if target_col_detected and target_col_detected in df.columns:
            hue_col = target_col_detected
            print(f"  Using '{hue_col}' for coloring in pairplot")
        else:
            hue_col = None
        
        g = sns.pairplot(df[sample_cols + ([hue_col] if hue_col else [])], 
                        hue=hue_col if hue_col else None,
                        diag_kind='kde', 
                        plot_kws={'alpha': 0.6, 's': 20},
                        height=2)
        g.fig.suptitle(f"Pairplot of Key Numerical Features", y=1.02, fontsize=12)
        plt.show()
    
    # === 7. Feature Importance Analysis ===
    print("\n" + "="*60)
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    target_col_detected = detect_target_column(df, target_col)
    
    if target_col_detected:
        analyze_feature_importance(df, target_col_detected, top_n=15)
    else:
        print("\nℹ No suitable target column detected for feature importance analysis.")
        print("  You can specify a target column using the 'target_col' parameter.")
        
        # Show all columns for user reference
        print("\n📋 Available columns:")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            unique = df[col].nunique()
            print(f"  {i:2d}. {col:<25} {dtype:<10} ({unique} unique values)")
    
    # === FINAL SUMMARY ===
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\n📊 Dataset Summary:")
    print(f"  Total features: {len(df.columns)}")
    print(f"    - Numerical: {len(num_cols)}")
    print(f"    - Categorical: {len(cat_cols)}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Missing values: {missing_total:,} ({missing_percent:.1f}%)")
    
    if target_col_detected:
        print(f"\n🎯 Target column: '{target_col_detected}'")
        target_unique = df[target_col_detected].nunique()
        print(f"  Unique values: {target_unique}")
        if target_unique <= 10:
            print(f"  Value counts: {df[target_col_detected].value_counts().to_dict()}")
    
    print(f"\n💡 Recommendations:")
    if missing_percent > 5:
        print("  • Consider imputing missing values")
    if len(num_cols) > 20:
        print("  • Consider dimensionality reduction for numerical features")
    if len(cat_cols) > 0:
        print("  • Encode categorical variables before modeling")
    
    print("\n✨ All visualizations generated successfully!")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Example usage
    final_file = "feature_outputs/step4_selected.csv"  # Adjust as needed
    
    # Option 1: Auto-detect target
    visualize_pipeline_results(final_file)
    
    # Option 2: Specify target column
    # visualize_pipeline_results(final_file, target_col=None)  # Auto-detect target
    
    # Option 3: Use with any CSV file
    # visualize_pipeline_results("path/to/your/dataset.csv", target_col='your_target')