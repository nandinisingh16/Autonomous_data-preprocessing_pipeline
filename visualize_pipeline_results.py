"""
File: visualize_pipeline_results.py
Purpose: Visualize results from the Autonomous Data Preprocessing Pipeline
Author: Raj Nandini Singh
Date: 2025-11-03
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# === CONFIG ===
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
sns.set(font_scale=1.1)

# === LOAD FINAL DATA ===
final_file = "feature_outputs/step4_selected.csv"  # adjust if needed
if not os.path.exists(final_file):
    raise FileNotFoundError(f"❌ {final_file} not found. Please run pipeline first.")

df = pd.read_csv(final_file)
print(f"✅ Loaded {final_file} with shape {df.shape}")
print("Columns:", df.columns.tolist())

# === QUICK OVERVIEW ===
print("\n🔍 Data Summary:")
print(df.info())
print("\n📊 Missing Values:")
print(df.isna().sum())

# === 1️Missing Value Heatmap ===
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='Reds')
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.show()

# === 2️Correlation Heatmap (Numerical) ===
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 1:
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[num_cols].corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Not enough numerical columns for correlation heatmap.")

# === 3️Distribution of Numeric Features ===
if len(num_cols) > 0:
    df[num_cols].hist(figsize=(14, 10), bins=25)
    plt.suptitle("Numeric Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

# === 4️Boxplots for Outlier Detection ===
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f"Outlier Check: {col}")
    plt.tight_layout()
    plt.show()

# === 5️Count Plots for Categorical Columns ===
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
for col in cat_cols:
    if df[col].nunique() <= 20:  # Avoid high-cardinality columns
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# === 6️Pairplot for Top Features ===
if len(num_cols) > 3:
    sample_cols = num_cols[:5]
    sns.pairplot(df[sample_cols])
    plt.suptitle("Pairplot of Key Numeric Features", y=1.02)
    plt.show()

# === 7️Feature Importance (if target exists) ===
target_col = None
for possible_target in ['Survived', 'target', 'label', 'Target']:
    if possible_target in df.columns:
        target_col = possible_target
        break

if target_col:
    from sklearn.ensemble import RandomForestClassifier
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.select_dtypes(include=[np.number]).fillna(0)

    if len(X.columns) > 0:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)[:10]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importance.values, y=importance.index, palette="viridis")
        plt.title(f"Top Feature Importances for Target '{target_col}'")
        plt.tight_layout()
        plt.show()
    else:
        print(" No numeric features available for feature importance.")

else:
    print("ℹ No target column found (e.g., 'Survived', 'target', 'label'). Skipping feature importance.")

print("\n Visualization complete — check the plots for insights!")
