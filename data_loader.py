"""
Data Loading Module
Handles loading and initial data exploration
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data from file
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def explore_data(df: pd.DataFrame) -> dict:
    """
    Perform initial data exploration
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    dict
        Dictionary containing data exploration results
    """
    exploration = {}
    
    # Basic info
    exploration['shape'] = df.shape
    exploration['columns'] = list(df.columns)
    exploration['dtypes'] = df.dtypes.to_dict()
    
    # Missing values
    exploration['missing_values'] = df.isnull().sum().to_dict()
    exploration['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()
    
    # Duplicates
    exploration['duplicate_rows'] = df.duplicated().sum()
    
    # Data types summary
    exploration['numerical_columns'] = df.select_dtypes(include=[np.number]).columns.tolist()
    exploration['categorical_columns'] = df.select_dtypes(include=['object']).columns.tolist()
    
    return exploration


def print_exploration(exploration: dict) -> None:
    """
    Print data exploration results
    
    Parameters:
    -----------
    exploration : dict
        Dictionary from explore_data function
    """
    print("\n" + "="*60)
    print("📊 DATA EXPLORATION REPORT")
    print("="*60)
    
    print(f"\n📐 Dataset Shape:")
    print(f"   Rows: {exploration['shape'][0]}")
    print(f"   Columns: {exploration['shape'][1]}")
    
    print(f"\n📋 Column Types:")
    print(f"   Numerical: {len(exploration['numerical_columns'])}")
    print(f"   Categorical: {len(exploration['categorical_columns'])}")
    
    print(f"\n❓ Missing Values:")
    missing = exploration['missing_values']
    if sum(missing.values()) == 0:
        print("   ✅ No missing values!")
    else:
        for col, count in missing.items():
            if count > 0:
                pct = exploration['missing_percentage'][col]
                print(f"   {col}: {count} ({pct:.2f}%)")
    
    print(f"\n🔄 Duplicate Rows: {exploration['duplicate_rows']}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if df is not None:
        exploration = explore_data(df)
        print_exploration(exploration)
