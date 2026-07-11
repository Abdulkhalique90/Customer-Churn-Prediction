"""
Data Preprocessing Module
Handles data cleaning, encoding, scaling, and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    strategy : str
        'drop' - remove rows with missing values
        'mean' - fill numerical with mean
        'median' - fill numerical with median
        
    Returns:
    --------
    pd.DataFrame
        Dataset with missing values handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    elif strategy == 'median':
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset without duplicates
    """
    df = df.copy()
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"✅ Removed {duplicates_count} duplicate rows")
    return df


def encode_categorical(df: pd.DataFrame, target_column: str = 'Churn') -> tuple:
    """
    Encode categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_column : str
        Name of target column
        
    Returns:
    --------
    tuple
        (processed_df, encoders_dict)
    """
    df = df.copy()
    encoders = {}
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col == target_column:
            # Binary encoding for target
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
        else:
            # One-hot encoding for features
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    return df, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(df: pd.DataFrame, target_column: str = 'Churn', 
                   test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    target_column : str
        Name of target column
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing train/test splits and preprocessing info
    """
    print("\n" + "="*60)
    print("🔧 DATA PREPROCESSING")
    print("="*60)
    
    # Step 1: Handle missing values
    print("\n1️⃣ Handling missing values...")
    df = handle_missing_values(df, strategy='median')
    print("   ✅ Missing values handled")
    
    # Step 2: Remove duplicates
    print("\n2️⃣ Removing duplicates...")
    df = handle_duplicates(df)
    
    # Step 3: Separate features and target
    print("\n3️⃣ Separating features and target...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Remove non-predictive columns
    cols_to_drop = ['customerID'] if 'customerID' in X.columns else []
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Step 4: Encode categorical variables
    print("\n4️⃣ Encoding categorical variables...")
    X, encoders = encode_categorical(X, target_column=None)
    
    # Encode target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        encoders[target_column] = le
    
    print(f"   Features after encoding: {X.shape}")
    
    # Step 5: Train-test split
    print(f"\n5️⃣ Splitting data ({100-test_size*100:.0f}% train, {test_size*100:.0f}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set: {X_test.shape}")
    
    # Step 6: Scale features
    print("\n6️⃣ Scaling numerical features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("   ✅ Features scaled using StandardScaler")
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60 + "\n")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_original': X_train,
        'X_test_original': X_test,
        'feature_names': list(X.columns),
        'encoders': encoders,
        'scaler': scaler
    }


if __name__ == "__main__":
    from data_loader import load_data
    
    # Example usage
    df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if df is not None:
        preprocessed = preprocess_data(df)
        print(f"Final training shape: {preprocessed['X_train'].shape}")
        print(f"Final testing shape: {preprocessed['X_test'].shape}")
