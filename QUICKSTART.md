# Quick Start Guide

## 📦 Installation

### 1. Download Kaggle Dataset
```bash
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place the CSV file in: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Run Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
Then open: http://localhost:8501

### Option 2: Run Jupyter Notebooks
```bash
jupyter notebook
# Run notebooks in order:
# 1. notebooks/01_exploratory_data_analysis.ipynb
# 2. notebooks/02_model_training.ipynb
```

### Option 3: Use Python Scripts Directly
```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_trainer import train_models
from src.evaluation import evaluate_models

# Load and process
df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
preprocessed = preprocess_data(df)

# Train and evaluate
models, trainer = train_models(
    preprocessed['X_train'], 
    preprocessed['y_train']
)
evaluator, comparison = evaluate_models(
    models,
    preprocessed['X_test'],
    preprocessed['y_test']
)
```

---

## 📊 Dashboard Features

1. **🏠 Home** - Project overview and statistics
2. **📊 Dataset Explorer** - Browse and download data
3. **📈 EDA Dashboard** - Interactive visualizations
4. **🤖 Model Training** - Train all 6 ML models
5. **📉 Model Comparison** - Compare model metrics
6. **🔮 Predict Churn** - Make predictions on new data

---

## 📁 Project Structure

```
customer-churn-prediction/
├── data/                    # Dataset directory
├── notebooks/              # Jupyter notebooks
├── src/                    # Python modules
│   ├── data_loader.py     # Load and explore data
│   ├── preprocessing.py    # Data cleaning
│   ├── model_trainer.py   # Train ML models
│   ├── evaluation.py      # Evaluate models
│   └── utils.py           # Utility functions
├── models/                 # Saved trained models
├── visuals/               # Generated plots
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Dependencies
└── README.md             # Full documentation
```

---

## 🤖 Available Models

1. **Logistic Regression** - Fast baseline
2. **Decision Tree** - Interpretable
3. **Random Forest** - Ensemble
4. **KNN** - Distance-based
5. **SVM** - Maximum margin
6. **XGBoost** - Best performer

---

## 📊 Key Metrics

- **Accuracy**: Overall correctness
- **Precision**: False positive rate
- **Recall**: False negative rate
- **F1-Score**: Balanced measure
- **ROC-AUC**: Classification capability

---

## 🎯 Next Steps

1. Download dataset from Kaggle
2. Install requirements: `pip install -r requirements.txt`
3. Run dashboard: `streamlit run app.py`
4. Explore data and train models
5. Deploy to Streamlit Cloud

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)
```bash
git push origin main
# Go to: https://streamlit.io/cloud
# Connect GitHub repo and deploy
```

### Alternative Deployments
- **Render**: render.com
- **Hugging Face**: huggingface.co/spaces
- **Heroku**: heroku.com

---

## 📞 Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review notebook comments
3. Check Streamlit documentation

---

**Happy predicting! 🎯**
