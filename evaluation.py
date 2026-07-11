"""
Model Evaluation Module
Evaluates and compares model performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
        self.predictions = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """
        Evaluate a single model
        
        Parameters:
        -----------
        model : object
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.results[model_name] = metrics
        self.predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return metrics
    
    def evaluate_all_models(self, models: dict, X_test, y_test):
        """
        Evaluate all models
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test target
            
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60 + "\n")
        
        for name, model in models.items():
            print(f"Evaluating {name}...", end=" ")
            try:
                self.evaluate_model(model, X_test, y_test, name)
                print("✅")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        # Create comparison dataframe
        comparison_df = self.get_comparison_dataframe()
        
        print("\n" + "="*60)
        print("📈 MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison_df.to_string())
        print("="*60 + "\n")
        
        return comparison_df
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get comparison dataframe of all models
        
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        data = []
        for model_name, metrics in self.results.items():
            data.append({
                'Model': metrics['Model'],
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1-Score': f"{metrics['F1-Score']:.4f}",
                'ROC-AUC': f"{metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] is not None else "N/A"
            })
        
        return pd.DataFrame(data)
    
    def get_best_model(self) -> str:
        """
        Get name of best performing model
        
        Returns:
        --------
        str
            Name of best model (by F1-Score)
        """
        best_model = max(self.results.items(), key=lambda x: x[1]['F1-Score'])
        return best_model[0]
    
    def plot_confusion_matrices(self, figsize=(15, 10)):
        """
        Plot confusion matrices for all models
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['Confusion Matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, figsize=(12, 8)):
        """
        Plot ROC curves for all models
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, preds in self.predictions.items():
            if preds['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(preds['y_test'], preds['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        return fig
    
    def plot_metrics_comparison(self, figsize=(14, 6)):
        """
        Plot comparison of metrics across models
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        data = []
        for model_name, metrics in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score']
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('Model')
        
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.legend(title='Metrics', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig


def evaluate_models(models: dict, X_test, y_test):
    """
    Main function to evaluate all models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test target
        
    Returns:
    --------
    tuple
        (evaluator_object, comparison_dataframe)
    """
    evaluator = ModelEvaluator()
    comparison = evaluator.evaluate_all_models(models, X_test, y_test)
    
    return evaluator, comparison


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    from model_trainer import train_models
    
    # Example usage
    df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if df is not None:
        preprocessed = preprocess_data(df)
        X_train = preprocessed['X_train']
        X_test = preprocessed['X_test']
        y_train = preprocessed['y_train']
        y_test = preprocessed['y_test']
        
        models, trainer = train_models(X_train, y_train)
        evaluator, comparison = evaluate_models(models, X_test, y_test)
        
        # Generate plots
        evaluator.plot_confusion_matrices()
        evaluator.plot_roc_curves()
        evaluator.plot_metrics_comparison()
