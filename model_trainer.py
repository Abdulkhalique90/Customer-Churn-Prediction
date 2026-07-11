"""
Model Training Module
Trains multiple machine learning models for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings

warnings.filterwarnings('ignore')


class ChurnModelTrainer:
    """Train and manage multiple churn prediction models"""
    
    def __init__(self):
        """Initialize model trainer with model configurations"""
        self.models = {}
        self.trained_models = {}
        self.best_models = {}
        
        # Define model configurations
        self.model_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {'n_estimators': [50, 100], 'max_depth': [10, 15], 'min_samples_split': [2, 5]}
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {'n_neighbors': [3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
            },
            'SVM': {
                'model': SVC(kernel='linear', probability=False, random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1, 10], 'kernel': ['linear']}
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, n_jobs=-1),
                'params': {'n_estimators': [50, 100], 'max_depth': [5, 7], 'learning_rate': [0.01, 0.1]}
            }
        }
    
    def train_basic_models(self, X_train, y_train):
        """
        Train basic models without hyperparameter tuning
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
            
        Returns:
        --------
        dict
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("🤖 TRAINING BASIC MODELS")
        print("="*60 + "\n")
        
        for name, config in self.model_configs.items():
            print(f"Training {name}...", end=" ")
            try:
                model = config['model']
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Get cross-validation score
                cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
                print(f"✅ (CV Score: {cv_score:.4f})")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        return self.trained_models
    
    def train_with_hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """
        Train models with hyperparameter tuning using GridSearchCV
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Dictionary of best models after tuning
        """
        print("\n" + "="*60)
        print("🎯 HYPERPARAMETER TUNING")
        print("="*60 + "\n")
        
        for name, config in self.model_configs.items():
            print(f"Tuning {name}...", end=" ")
            try:
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                
                self.best_models[name] = grid_search.best_estimator_
                best_score = grid_search.best_score_
                
                print(f"✅ (Best CV Score: {best_score:.4f})")
                print(f"   Best params: {grid_search.best_params_}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        return self.best_models
    
    def save_models(self, directory: str = '../models'):
        """
        Save trained models to disk
        
        Parameters:
        -----------
        directory : str
            Directory to save models
        """
        print(f"\n💾 Saving models to {directory}...")
        
        models_to_save = self.best_models if self.best_models else self.trained_models
        
        for name, model in models_to_save.items():
            filename = f"{directory}/{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filename)
            print(f"   ✅ {name} saved")
    
    def load_models(self, directory: str = '../models'):
        """
        Load trained models from disk
        
        Parameters:
        -----------
        directory : str
            Directory containing saved models
            
        Returns:
        --------
        dict
            Dictionary of loaded models
        """
        import os
        from pathlib import Path
        
        print(f"\n📂 Loading models from {directory}...")
        
        models = {}
        model_dir = Path(directory)
        
        if not model_dir.exists():
            print(f"   ❌ Directory not found: {directory}")
            return models
        
        for file in model_dir.glob('*.pkl'):
            name = file.stem.replace('_', ' ').title()
            try:
                model = joblib.load(file)
                models[name] = model
                print(f"   ✅ {name} loaded")
            except Exception as e:
                print(f"   ❌ Error loading {name}: {str(e)}")
        
        return models
    
    def get_model(self, name: str):
        """
        Get a specific model
        
        Parameters:
        -----------
        name : str
            Model name
            
        Returns:
        --------
        object
            Trained model or None if not found
        """
        if name in self.best_models:
            return self.best_models[name]
        elif name in self.trained_models:
            return self.trained_models[name]
        else:
            return None


def train_models(X_train, y_train, tune: bool = False):
    """
    Main function to train all models
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    tune : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    trainer = ChurnModelTrainer()
    
    # Train basic models
    trainer.train_basic_models(X_train, y_train)
    
    # Optionally tune hyperparameters
    if tune:
        trainer.train_with_hyperparameter_tuning(X_train, y_train)
        trained = trainer.best_models
    else:
        trained = trainer.trained_models
    
    return trained, trainer


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    
    # Example usage
    df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if df is not None:
        preprocessed = preprocess_data(df)
        X_train = preprocessed['X_train']
        y_train = preprocessed['y_train']
        
        models, trainer = train_models(X_train, y_train, tune=False)
        print(f"\n✅ Trained {len(models)} models")
