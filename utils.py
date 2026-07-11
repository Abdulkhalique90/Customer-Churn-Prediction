"""
Utility Functions
Helper functions for the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def set_plot_style():
    """Set consistent plot styling"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def create_directory_if_not_exists(directory: str):
    """
    Create directory if it doesn't exist
    
    Parameters:
    -----------
    directory : str
        Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_dataframe_to_csv(df: pd.DataFrame, filepath: str):
    """
    Save dataframe to CSV with timestamp
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str
        Output file path
    """
    create_directory_if_not_exists(str(Path(filepath).parent))
    df.to_csv(filepath, index=False)
    print(f"✅ Saved to {filepath}")


def plot_distribution(df: pd.DataFrame, column: str, figsize=(10, 6)):
    """
    Plot distribution of a numerical column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    df[column].hist(bins=30, ax=ax, edgecolor='black')
    ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(alpha=0.3)
    return fig


def plot_categorical(df: pd.DataFrame, column: str, target: str = None, figsize=(10, 6)):
    """
    Plot distribution of categorical column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name
    target : str
        Target column for hue
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if target:
        sns.countplot(data=df, x=column, hue=target, ax=ax)
        ax.set_title(f'{column} vs {target}', fontsize=14, fontweight='bold')
    else:
        sns.countplot(data=df, x=column, ax=ax)
        ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, figsize=(12, 10)):
    """
    Plot correlation heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (numerical columns only)
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def get_feature_importance(model, feature_names: list, top_n: int = 10):
    """
    Get feature importance from tree-based models
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df.head(top_n)


def plot_feature_importance(importance_df: pd.DataFrame, figsize=(10, 6)):
    """
    Plot feature importance
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title('Top Features by Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    return fig


def print_summary_statistics(df: pd.DataFrame, column: str):
    """
    Print summary statistics for a column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name
    """
    print(f"\n📊 Summary Statistics: {column}")
    print("="*50)
    print(df[column].describe())
    print("="*50 + "\n")


def get_churn_rate(df: pd.DataFrame, column: str, target: str = 'Churn'):
    """
    Calculate churn rate for a categorical feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Feature column
    target : str
        Target column
        
    Returns:
    --------
    pd.DataFrame
        Churn rate by category
    """
    churn_rate = df.groupby(column)[target].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    return pd.DataFrame({
        column: churn_rate.index,
        'Churn_Rate_%': churn_rate.values
    })


if __name__ == "__main__":
    # Example usage
    set_plot_style()
    print("✅ Utility functions loaded")
