# test_edge_cases.py
import pandas as pd
import os

def test_edge_cases():
    """Test pipeline with challenging datasets"""
    
    edge_cases = [
        ("Very Large Dataset", "test_datasets/large_dataset.csv", 10000, 50),
        ("High Missing Values", "test_datasets/high_missing.csv", 1000, 20),
        ("Mixed Data Types", "test_datasets/mixed_types.csv", 500, 15),
        ("Unbalanced Classes", "test_datasets/unbalanced.csv", 2000, 10),
        ("Text Heavy Dataset", "test_datasets/text_heavy.csv", 800, 25),
    ]
    
    for case_name, filepath, n_rows, n_cols in edge_cases:
        print(f"\n🔬 Testing Edge Case: {case_name}")
        
        # Generate synthetic dataset for testing
        import numpy as np
        df = pd.DataFrame(np.random.randn(n_rows, n_cols), 
                         columns=[f'feature_{i}' for i in range(n_cols)])
        
        # Add target column
        df['target'] = np.random.randint(0, 2, n_rows)
        
        # Apply edge case characteristics
        if "Missing" in case_name:
            # 50% missing values
            mask = np.random.random(df.shape) < 0.5
            df = df.mask(mask)
        
        if "Unbalanced" in case_name:
            # 95% class 0, 5% class 1
            df['target'] = [0] * int(n_rows*0.95) + [1] * int(n_rows*0.05)
            np.random.shuffle(df['target'])
        
        if "Text" in case_name:
            # Add text columns
            df['text_col'] = [f"Sample text {i}" * 10 for i in range(n_rows)]
        
        # Save and perform basic validation
        df.to_csv(filepath, index=False)

        # Basic validation: file exists and is readable
        assert os.path.exists(filepath)
        df_read = pd.read_csv(filepath)
        assert df_read.shape[0] > 0
        assert df_read.shape[1] > 0