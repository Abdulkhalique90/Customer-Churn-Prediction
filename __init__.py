"""
Customer Churn Prediction Project
Main module for churn prediction pipeline
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from .data_loader import load_data
from .preprocessing import preprocess_data
from .model_trainer import train_models
from .evaluation import evaluate_models

__all__ = [
    "load_data",
    "preprocess_data",
    "train_models",
    "evaluate_models"
]
