# test_datasets_download.py
import pandas as pd
from sklearn.datasets import fetch_openml, load_iris, load_diabetes, load_wine
import os

os.makedirs("test_datasets", exist_ok=True)

# 1. Titanic (already have)
titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
titanic.to_csv("test_datasets/titanic.csv", index=False)



# 4. Diabetes Dataset
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(diabetes.data, columns=[f'feature_{i}' for i in range(diabetes.data.shape[1])])
diabetes_df['target'] = diabetes.target
diabetes_df.to_csv("test_datasets/diabetes.csv", index=False)

# 5. Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target
cancer_df.to_csv("test_datasets/B_cancer.csv", index=False)

# 6. Add missing values for testing
import numpy as np
np.random.seed(42)
cancer_with_missing = cancer_df.copy()
for col in cancer_with_missing.columns[:5]:
    mask = np.random.random(len(cancer_with_missing)) < 0.1
    cancer_with_missing.loc[mask, col] = np.nan
cancer_with_missing.to_csv("test_datasets/M_cancer.csv", index=False)

print("Test datasets created!")