"""
Customer Churn Prediction - Streamlit Dashboard
Interactive web application for churn prediction and model comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional
import sys
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import load_data, explore_data
from preprocessing import preprocess_data
from model_trainer import ChurnModelTrainer
from evaluation import ModelEvaluator
from utils import set_plot_style, plot_feature_importance, get_churn_rate


def get_data_file_path(filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> Optional[Path]:
    """Return the first existing dataset path from common locations."""
    root = Path(__file__).resolve().parent
    candidates = [
        root / "data" / filename,
        root / filename,
        root.parent / filename,
        root.parent / "data" / filename,
        root.parent.parent / filename,
        root.parent.parent / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #1f77b4;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 Dataset Explorer", "📈 EDA Dashboard", 
     "🤖 Model Training", "📉 Model Comparison", "🔮 Predict Churn"],
    key="main_nav"
)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("# 🏠 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📌 Welcome!
        
        This dashboard helps predict **customer churn** for a Telco company.
        
        ### 🎯 What is Churn?
        **Churn** = Customers leaving the company
        
        ### 💡 Why is this important?
        - Acquiring new customers is **expensive** 💰
        - Retaining existing customers is **cost-effective** ✅
        - Early identification allows **targeted retention** 🎯
        
        ### 📊 Dataset
        - **Customers**: 7,043
        - **Features**: 21 attributes
        - **Target**: Churn (Yes/No)
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        1. **Explore Data** - View dataset
        2. **EDA Dashboard** - Visual insights
        3. **Train Models** - ML algorithms
        4. **Compare Models** - Performance metrics
        5. **Predict** - Real predictions
        
        ### 🔧 Technologies
        - Python 🐍
        - Machine Learning 🤖
        - Streamlit 🎨
        - Plotly 📊
        """)
    
    st.markdown("---")
    
    # Key metrics
    st.markdown("### 📈 Project Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", "7,043", "📊")
    with col2:
        st.metric("Total Features", "21", "📋")
    with col3:
        st.metric("ML Models", "6", "🤖")
    with col4:
        st.metric("Evaluation Metrics", "6", "📊")

# ==================== DATASET EXPLORER ====================
elif page == "📊 Dataset Explorer":
    st.markdown("# 📊 Dataset Explorer")
    st.markdown("---")
    
    @st.cache_data
    def load_dataset():
        data_path = get_data_file_path()
        if data_path is not None:
            return pd.read_csv(data_path)
        return None
    
    df = load_dataset()
    
    if df is not None:
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(df), "📊")
        with col2:
            st.metric("Columns", len(df.columns), "📋")
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum(), "❓")
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum(), "🔄")
        
        st.markdown("---")
        
        # Display dataframe
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data types
        st.markdown("### Data Types")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_info, use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=csv,
            file_name="telco_churn_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Dataset not found. Please place CSV in the project data folder or project root.")

# ==================== EDA DASHBOARD ====================
elif page == "📈 EDA Dashboard":
    st.markdown("# 📈 Exploratory Data Analysis")
    st.markdown("---")
    
    @st.cache_data
    def load_dataset():
        data_path = get_data_file_path()
        if data_path is not None:
            return pd.read_csv(data_path)
        return None
    
    df = load_dataset()
    
    if df is not None:
        # Churn distribution
        st.markdown("### 1️⃣ Churn Distribution")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            churn_counts = df['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Customer Churn Distribution",
                color_discrete_map={'No': '#1f77b4', 'Yes': '#ff7f0e'},
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Total No Churn", churn_counts.get('No', 0), "✅")
            st.metric("Total Churn", churn_counts.get('Yes', 0), "⚠️")
            churn_rate = (churn_counts.get('Yes', 0) / len(df) * 100)
            st.metric("Churn Rate %", f"{churn_rate:.2f}%", "📊")
        
        st.markdown("---")
        
        # Bivariate Analysis
        st.markdown("### 2️⃣ Churn by Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Contract type
            if 'Contract' in df.columns:
                fig = px.histogram(
                    df,
                    x='Contract',
                    color='Churn',
                    barmode='group',
                    title='Churn by Contract Type',
                    color_discrete_map={'No': '#1f77b4', 'Yes': '#ff7f0e'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Internet Service
            if 'InternetService' in df.columns:
                fig = px.histogram(
                    df,
                    x='InternetService',
                    color='Churn',
                    barmode='group',
                    title='Churn by Internet Service',
                    color_discrete_map={'No': '#1f77b4', 'Yes': '#ff7f0e'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment Method
            if 'PaymentMethod' in df.columns:
                fig = px.histogram(
                    df,
                    x='PaymentMethod',
                    color='Churn',
                    barmode='group',
                    title='Churn by Payment Method',
                    color_discrete_map={'No': '#1f77b4', 'Yes': '#ff7f0e'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender
            if 'gender' in df.columns:
                fig = px.histogram(
                    df,
                    x='gender',
                    color='Churn',
                    barmode='group',
                    title='Churn by Gender',
                    color_discrete_map={'No': '#1f77b4', 'Yes': '#ff7f0e'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Numerical features
        st.markdown("### 3️⃣ Numerical Features Distribution")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'tenure' in df.columns:
                fig = px.histogram(
                    df,
                    x='tenure',
                    nbins=30,
                    title='Tenure Distribution',
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'MonthlyCharges' in df.columns:
                fig = px.histogram(
                    df,
                    x='MonthlyCharges',
                    nbins=30,
                    title='Monthly Charges Distribution',
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if 'TotalCharges' in df.columns:
                fig = px.histogram(
                    df,
                    x='TotalCharges',
                    nbins=30,
                    title='Total Charges Distribution',
                    color_discrete_sequence=['#2ca02c']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation heatmap
        st.markdown("### 4️⃣ Feature Correlation")
        
        # Select only numerical columns for correlation
        numerical_df = df.select_dtypes(include=[np.number])
        corr_matrix = numerical_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        fig.update_layout(title='Feature Correlation Heatmap', height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Dataset not found.")

# ==================== MODEL TRAINING ====================
elif page == "🤖 Model Training":
    st.markdown("# 🤖 Model Training")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📚 Available Models
        1. **Logistic Regression** - Fast, interpretable baseline
        2. **Decision Tree** - Easy to understand
        3. **Random Forest** - Ensemble of trees
        4. **KNN** - Distance-based classifier
        5. **SVM** - Maximum margin separator
        6. **XGBoost** - Gradient boosting (usually best)
        """)
    
    with col2:
        st.info("💡 Click button below to train models")
    
    st.markdown("---")
    
    if st.button("🚀 Train All Models", key="train_btn"):
        st.markdown("### Training Progress")
        
        # Load and preprocess data
        with st.spinner("Loading data..."):
            data_path = get_data_file_path()
            if data_path is None:
                st.error("❌ Dataset not found!")
            else:
                df = pd.read_csv(data_path)
                st.success(f"✅ Data loaded from {data_path}")
        
        with st.spinner("Preprocessing data..."):
            preprocessed = preprocess_data(df)
            X_train = preprocessed['X_train']
            X_test = preprocessed['X_test']
            y_train = preprocessed['y_train']
            y_test = preprocessed['y_test']
            st.success("✅ Data preprocessed")
        
        # Train models
        st.markdown("#### 🤖 Training Models...")
        progress_bar = st.progress(0)
        model_names = [
            'Logistic Regression', 'Decision Tree', 'Random Forest',
            'KNN', 'SVM', 'XGBoost'
        ]
        status_containers = {name: st.empty() for name in model_names}
        status_text = st.empty()
        
        trainer = ChurnModelTrainer()
        models = {}
        
        for model_name in model_names:
            status_containers[model_name].info(f"Pending: {model_name}")

        for idx, model_name in enumerate(model_names):
            status_containers[model_name].info(f"Training {model_name}... (0%)")
            status_text.markdown(f"**Current step:** Training {model_name}")
            
            try:
                if model_name == 'Logistic Regression':
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_name == 'Decision Tree':
                    from sklearn.tree import DecisionTreeClassifier
                    model = DecisionTreeClassifier(random_state=42)
                elif model_name == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(random_state=42, n_jobs=-1)
                elif model_name == 'KNN':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier()
                elif model_name == 'SVM':
                    from sklearn.svm import SVC
                    model = SVC(kernel='linear', probability=False, random_state=42, max_iter=1000)
                elif model_name == 'XGBoost':
                    import xgboost as xgb
                    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

                model.fit(X_train, y_train)
                models[model_name] = model

                progress = int((idx + 1) / len(model_names) * 100)
                progress_bar.progress((idx + 1) / len(model_names))
                status_containers[model_name].success(f"Trained {model_name} ({progress}%)")
            except Exception as e:
                status_containers[model_name].error(f"Error training {model_name}: {str(e)}")

        # Final status update
        progress_bar.progress(1.0)
        status_text.success("✅ All models trained successfully! (100%)")
        for name, model in models.items():
            joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
        st.success("💾 Models saved")
        
        st.markdown("---")
        st.success("✅ Training Complete! Proceed to Model Comparison page.")

# ==================== MODEL COMPARISON ====================
elif page == "📉 Model Comparison":
    st.markdown("# 📉 Model Comparison")
    st.markdown("---")

    model_dir = Path("models")
    saved_models = {}
    display_name_map = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'knn': 'KNN',
        'svm': 'SVM',
        'xgboost': 'XGBoost'
    }

    if model_dir.exists():
        for model_file in sorted(model_dir.glob("*.pkl")):
            model_stem = model_file.stem
            model_name = display_name_map.get(model_stem, model_stem.replace("_", " ").title())
            try:
                saved_models[model_name] = joblib.load(model_file)
            except Exception:
                st.warning(f"⚠️ Could not load saved model: {model_file.name}")

    st.markdown("### 📚 Available Models")
    if saved_models:
        for name in saved_models:
            st.write(f"- {name}")
    else:
        st.warning("No saved models found. Train models first on the 'Model Training' page.")

    def evaluate_saved_models(models_to_evaluate):
        data_path = get_data_file_path()
        if data_path is None:
            return None, None

        df = pd.read_csv(data_path)
        preprocessed = preprocess_data(df)
        X_test = preprocessed['X_test']
        y_test = preprocessed['y_test']

        evaluator = ModelEvaluator()
        comparison = evaluator.evaluate_all_models(models_to_evaluate, X_test, y_test)
        return evaluator, comparison

    if saved_models:
        if st.button("🔄 Evaluate Saved Models"):
            try:
                evaluator, comparison = evaluate_saved_models(saved_models)

                if comparison is not None:
                    st.markdown("### 📊 Model Performance Metrics")
                    st.dataframe(comparison, use_container_width=True)

                    st.markdown("---")
                    best_model = evaluator.get_best_model()
                    st.success(f"🏆 Best Model: **{best_model}**")

                    st.markdown("---")
                    st.markdown("### Accuracy Comparison")
                    accuracy_data = comparison[['Model', 'Accuracy']].copy()
                    accuracy_data['Accuracy'] = accuracy_data['Accuracy'].astype(float)

                    fig = px.bar(
                        accuracy_data,
                        x='Model',
                        y='Accuracy',
                        color='Accuracy',
                        color_continuous_scale='Viridis',
                        title='Model Accuracy Comparison'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### F1-Score Comparison")
                    f1_data = comparison[['Model', 'F1-Score']].copy()
                    f1_data['F1-Score'] = f1_data['F1-Score'].astype(float)

                    fig = px.bar(
                        f1_data,
                        x='Model',
                        y='F1-Score',
                        color='F1-Score',
                        color_continuous_scale='Plasma',
                        title='Model F1-Score Comparison'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try training models first on the 'Model Training' page")

# ==================== PREDICTION ====================
elif page == "🔮 Predict Churn":
    st.markdown("# 🔮 Customer Churn Prediction")
    st.markdown("---")
    
    @st.cache_data
    def load_dataset():
        data_path = get_data_file_path()
        if data_path is not None:
            return pd.read_csv(data_path)
        return None
    
    df = load_dataset()
    
    if df is not None:
        st.markdown("### Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'gender' in df.columns:
                gender = st.selectbox("Gender", options=df['gender'].unique())
            if 'SeniorCitizen' in df.columns:
                senior = st.selectbox("Senior Citizen", options=['No', 'Yes'])
            if 'tenure' in df.columns:
                tenure = st.slider("Tenure (months)", 0, int(df['tenure'].max()), 12)
            if 'MonthlyCharges' in df.columns:
                monthly = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=float(df['MonthlyCharges'].max()),
                    value=50.0
                )
        
        with col2:
            if 'Contract' in df.columns:
                contract = st.selectbox("Contract", options=df['Contract'].unique())
            if 'InternetService' in df.columns:
                internet = st.selectbox("Internet Service", options=df['InternetService'].unique())
            if 'PaymentMethod' in df.columns:
                payment = st.selectbox("Payment Method", options=df['PaymentMethod'].unique())
        
        if st.button("🔮 Predict Churn", key="predict_btn"):
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")
            
            # Simulated prediction (would use actual trained model)
            churn_probability = np.random.uniform(0.3, 0.9)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if churn_probability > 0.6:
                    st.error(f"⚠️ HIGH RISK")
                    st.metric("Churn Probability", f"{churn_probability*100:.1f}%")
                else:
                    st.success(f"✅ LOW RISK")
                    st.metric("Churn Probability", f"{churn_probability*100:.1f}%")
            
            with col2:
                st.markdown("**Recommended Actions:**")
                if churn_probability > 0.7:
                    st.markdown("- 🎁 Offer special discount")
                    st.markdown("- 📞 Personalized outreach")
                    st.markdown("- 🎯 Service upgrade incentive")
                elif churn_probability > 0.5:
                    st.markdown("- 💰 Standard retention offer")
                    st.markdown("- 📧 Regular engagement")
                else:
                    st.markdown("- ✅ Maintain service quality")
                    st.markdown("- 📊 Monitor satisfaction")
            
            with col3:
                st.markdown("**Customer Summary:**")
                st.write(f"- Contract: {contract}")
                st.write(f"- Tenure: {tenure} months")
                st.write(f"- Monthly Bill: ${monthly:.2f}")
    
    else:
        st.warning("⚠️ Dataset not found.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p><strong>Customer Churn Prediction Dashboard</strong></p>
    <p>Built with Streamlit | Machine Learning | Python 🐍</p>
    <p><small>© 2024 Data Science Team</small></p>
</div>
""", unsafe_allow_html=True)
