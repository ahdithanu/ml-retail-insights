# =======================
# Crash guards: MUST be first
# =======================
import os
os.environ["MPLBACKEND"] = "Agg"            # non-GUI backend for matplotlib
os.environ["OMP_NUM_THREADS"] = "1"         # avoid OpenMP/libomp explosions on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # tolerate MKL/libomp duplicates (conda)

import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# =======================
# Standard imports
# =======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Core ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, silhouette_score, accuracy_score,
    precision_score, recall_score, f1_score
)

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# other stdlib
import sys
import traceback
from io import StringIO
import base64

# =======================
# Streamlit page config
# =======================
st.set_page_config(
    page_title="Customer Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Theme / CSS
# =======================
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        text-align: left;
        color: #1f77b4;
        margin: 0.25rem 0 0.75rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# Global state & colors
# =======================
PLOTLY_PALETTE = px.colors.qualitative.D3

def _palette(i: int) -> str:
    return PLOTLY_PALETTE[i % len(PLOTLY_PALETTE)]

# Persistent app-wide trained models
if "trained_models" not in st.session_state:
    # { model_name: {"model":..., "metrics":..., "y_test":..., "y_pred":..., "y_proba":..., "features":..., "scaler":...} }
    st.session_state.trained_models = {}

if "last_trained" not in st.session_state:
    st.session_state.last_trained = None  # string name of most recently trained model

# Persistent segmentation (KMeans) state
if "segmentation_state" not in st.session_state:
    # {"clusters": { 'labels', 'centers', 'scaler', 'features', 'k', 'data' } }
    st.session_state.segmentation_state = {"clusters": None}

# =======================
# Dashboard class
# =======================
class CustomerRetentionDashboard:
    def __init__(self):
        self.data = None
        self.processed_data = None
        # legacy members kept (but we use session_state for persistence)
        self.models = {}
        self.model_results = {}
        self.clusters = None

    # ---------- Data loading ----------
    def load_data(self):
        st.sidebar.header("📁 Data Upload")
        data_option = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload CSV"])

        if data_option == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                try:
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            self.data = pd.read_csv(uploaded_file, encoding=encoding)
                            st.sidebar.success(f"Loaded {len(self.data)} rows with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        st.sidebar.error("Could not read file with any standard encoding")
                        return False

                    if len(self.data) == 0:
                        st.sidebar.error("CSV file is empty")
                        return False
                    if len(self.data.columns) < 3:
                        st.sidebar.error("CSV must have at least 3 columns")
                        return False
                except Exception as e:
                    st.sidebar.error(f"Error loading file: {str(e)}")
                    return False
        else:
            try:
                self.data = self.generate_sample_data()
                st.sidebar.success("Using sample customer data")
            except Exception as e:
                st.sidebar.error(f"Error generating sample data: {str(e)}")
                return False
        return self.data is not None

    def generate_sample_data(self):
        np.random.seed(42)
        n_customers = 1000
        data = {
            'id': range(1, n_customers + 1),
            'age': np.random.randint(18, 70, n_customers),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_customors := n_customers, p=[0.45, 0.45, 0.1]),
            'income': np.random.randint(30000, 150000, n_customers),
            'spending_score': np.random.randint(1, 101, n_customers),
            'membership_years': np.random.randint(1, 11, n_customers),
            'purchase_frequency': np.random.randint(1, 51, n_customers),
            'preferred_category': np.random.choice(['Electronics','Clothing','Groceries','Sports','Home & Garden'], n_customers),
            'last_purchase_amount': np.round(np.random.uniform(10, 1000, n_customers), 2)
        }
        return pd.DataFrame(data)

    def preprocess_data(self):
        if self.data is None:
            return
        df = self.data.copy()
        # Feature engineering
        df['loyalty_score'] = (
            df['purchase_frequency'].rank(pct=True) +
            df['spending_score'].rank(pct=True)
        ) / 2

        # Simulated churn proxy for demo
        star_cutoff = df['loyalty_score'].quantile(0.95)
        df['is_star_customer'] = df['loyalty_score'] >= star_cutoff
        df['churn_risk'] = (
            (df['loyalty_score'] < 0.3) |
            (df['purchase_frequency'] < df['purchase_frequency'].quantile(0.2))
        ).astype(int)

        # Groups
        df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
        df['tenure_group'] = pd.cut(df['membership_years'], bins=[0,2,5,8,float('inf')], labels=['<2 yrs','2-5 yrs','5-8 yrs','8+ yrs'])

        self.processed_data = df

    # ---------- Summary ----------
    def summary_tab(self):
        st.markdown('<div class="main-header">📊 Executive Summary</div>', unsafe_allow_html=True)
        if self.processed_data is None:
            st.warning("Please load and process data first")
            return
        df = self.processed_data

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            churn_rate = (df['churn_risk'].sum() / len(df)) * 100
            st.metric("Churn Risk Rate", f"{churn_rate:.1f}%")
        with col3:
            star_customers = df['is_star_customer'].sum()
            star_rate = (star_customers / len(df)) * 100
            st.metric("Star Customers", f"{star_customers} ({star_rate:.1f}%)")
        with col4:
            st.metric("Avg Loyalty Score", f"{df['loyalty_score'].mean():.3f}")

        st.markdown("---")
        st.subheader("🔍 Key Insights")
        colA, colB = st.columns(2)
        with colA:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Customer Distribution**")
            cat_dist = df['preferred_category'].value_counts()
            fig_cat = px.bar(
                x=cat_dist.index, y=cat_dist.values,
                title="Customers by Preferred Category",
                color=cat_dist.index,
                color_discrete_sequence=PLOTLY_PALETTE
            )
            fig_cat.update_layout(showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with colB:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Loyalty Distribution**")
            fig_loyalty = px.histogram(
                df, x='loyalty_score', nbins=30,
                title="Loyalty Score Distribution",
                color_discrete_sequence=[_palette(1)]
            )
            st.plotly_chart(fig_loyalty, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("📋 Strategic Recommendations")
        for i, rec in enumerate([
            "Target customers with loyalty_score < 0.3 (high churn proxy) with retention offers.",
            "Consider higher-touch programs for long-tenure (8+ yrs) high spenders.",
            "Create premium loyalty tier for top 5% 'star customers' to lock in value.",
            "Monitor mid-income cohorts that show high variance in engagement.",
            "Expand high-engagement categories (Electronics, Sports)."
        ], 1):
            st.markdown(f"**{i}.** {rec}")

    # ---------- EDA ----------
    def eda_tab(self):
        st.markdown('<div class="main-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        if self.processed_data is None:
            st.warning("Please load and process data first")
            return
        df = self.processed_data

        st.subheader("📊 Data Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Missing Values:**", int(df.isnull().sum().sum()))
        with c2:
            st.write("**Numerical Columns:**", len(df.select_dtypes(include=[np.number]).columns))
            st.write("**Categorical Columns:**", len(df.select_dtypes(include=['object']).columns))

        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("📈 Distribution Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cc1, cc2 = st.columns(2)
        with cc1:
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
        with cc2:
            chart_type = st.selectbox("Chart type:", ["Histogram", "Box Plot"])
        if chart_type == "Histogram":
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}",
                               color_discrete_sequence=[_palette(2)])
        else:
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}",
                         color_discrete_sequence=[_palette(2)])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔗 Correlation Analysis")
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, title="Correlation Heatmap", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("🎯 Key Relationships")
        rr1, rr2 = st.columns(2)
        with rr1:
            x_var = st.selectbox("X-axis variable:", numeric_cols, key="x_var")
        with rr2:
            y_var = st.selectbox("Y-axis variable:", numeric_cols, key="y_var")
        fig_scatter = px.scatter(df, x=x_var, y=y_var, color='preferred_category',
                                 title=f"{y_var} vs {x_var}",
                                 color_discrete_sequence=PLOTLY_PALETTE)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------- Segmentation (KMeans) with persistence ----------
    def _get_cluster_store(self):
        return st.session_state.segmentation_state

    def segmentation_tab(self):
        st.markdown('<div class="main-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)
        st.caption("ℹ️ Run K-Means here; results persist while the app is open.")

        if self.processed_data is None:
            st.warning("Please load and process data first")
            return

        df = self.processed_data
        store = self._get_cluster_store()

        st.subheader("K-Means Configuration")
        numeric_features = [
            'age','income','spending_score','membership_years',
            'purchase_frequency','last_purchase_amount'
        ]
        c1, c2 = st.columns([2,1])
        with c1:
            selected_features = st.multiselect(
                "Select features for clustering:",
                numeric_features,
                default=['spending_score','purchase_frequency','income']
            )
        with c2:
            n_clusters = st.slider("Number of clusters (k):", 2, 8, 3)

        run_clicked = st.button("Run K-Means Clustering", use_container_width=True)

        if run_clicked:
            if len(selected_features) < 2:
                st.error("Please select at least 2 features for clustering")
            else:
                self.run_kmeans_clustering(df, selected_features, n_clusters, store=store)

        # Results
        if store.get('clusters') is None:
            st.info("No clustering results yet. Configure options above and click **Run K-Means Clustering**.")
            return

        # small badge
        st.success(f"Last run — k={store['clusters']['k']} on {', '.join(store['clusters']['features'])}")

        self.display_clustering_results(df, store['clusters']['features'], store=store)

    def run_kmeans_clustering(self, df, features, k, store=None):
        try:
            if store is None:
                store = self._get_cluster_store()
            if len(features) < 2:
                st.error("At least 2 features required for clustering")
                return
            if k < 2 or k > len(df):
                st.error("Invalid number of clusters")
                return

            X = df[features].copy()
            # handle missing
            if X.isnull().sum().sum() > 0:
                st.warning("Found missing values, filling with median")
                for col in X.columns:
                    if X[col].dtype in ['int64', 'float64']:
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        try:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                            X[col].fillna(X[col].median(), inplace=True)
                        except:
                            st.error(f"Cannot process non-numeric column: {col}")
                            return

            # constant columns
            for col in X.columns:
                if X[col].nunique() <= 1:
                    st.warning(f"Column {col} has constant values; clustering may not be meaningful")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                st.error("Scaling produced invalid values")
                return

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            unique_labels = len(np.unique(cluster_labels))
            if unique_labels < k:
                st.warning(f"Only {unique_labels} unique clusters found (expected {k})")

            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = cluster_labels
            store['clusters'] = {
                'labels': cluster_labels,
                'centers': kmeans.cluster_centers_,
                'scaler': scaler,
                'features': features,
                'k': k,
                'data': df_with_clusters
            }
            st.success(f"Clustering completed with {k} clusters!")
        except Exception as e:
            st.error(f"Unexpected error in clustering: {str(e)}")

    def display_clustering_results(self, df, features, store=None):
        if store is None or store.get('clusters') is None:
            st.info("No clustering to display yet.")
            return
        cluster_data = store['clusters']['data']

        st.subheader("📊 Cluster Summary")
        cluster_summary = cluster_data.groupby('Cluster')[features + ['loyalty_score']].agg(['mean','count']).round(2)
        st.dataframe(cluster_summary, use_container_width=True)

        st.subheader("📈 Cluster Visualization")
        if len(features) >= 2:
            fig = px.scatter(
                cluster_data, x=features[0], y=features[1],
                color='Cluster',
                title=f"Clusters: {features[1]} vs {features[0]}",
                color_discrete_sequence=PLOTLY_PALETTE
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("👤 Cluster Profiles")
        for cid in sorted(cluster_data['Cluster'].unique()):
            subset = cluster_data[cluster_data['Cluster'] == cid]
            with st.expander(f"Cluster {cid} ({len(subset)} customers)"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Avg Age", f"{subset['age'].mean():.1f}")
                    st.metric("Avg Income", f"${subset['income'].mean():,.0f}")
                with c2:
                    st.metric("Avg Spending Score", f"{subset['spending_score'].mean():.1f}")
                    st.metric("Avg Purchase Freq", f"{subset['purchase_frequency'].mean():.1f}")
                with c3:
                    st.metric("Avg Loyalty Score", f"{subset['loyalty_score'].mean():.3f}")
                    top_category = subset['preferred_category'].mode().iloc[0] if len(subset) > 0 else "N/A"
                    st.metric("Top Category", top_category)

    # ---------- Churn Models (persist results) ----------
    def churn_models_tab(self):
        st.markdown('<div class="main-header">🤖 Churn Prediction Models</div>', unsafe_allow_html=True)
        st.caption("ℹ️ Models you train here are available across other tabs until you reload the app.")
        if self.processed_data is None:
            st.warning("Please load and process data first")
            return

        df = self.processed_data
        st.sidebar.subheader("🛠️ Model Configuration")
        model_type = st.sidebar.selectbox(
            "Select Model:",
            ["Logistic Regression", "Random Forest", "Decision Tree",
             "Gradient Boosting", "SVM", "Neural Network (MLP)",
             "AdaBoost", "Naive Bayes", "K-Nearest Neighbors",
             "ElasticNet", "Ridge Regression", "Extra Trees"],
            index=0
        )
        available_features = ['age','income','spending_score','membership_years',
                              'purchase_frequency','last_purchase_amount','loyalty_score']
        selected_features = st.sidebar.multiselect(
            "Select features:", available_features,
            default=['spending_score','purchase_frequency','loyalty_score']
        )
        if len(selected_features) < 1:
            st.error("Please select at least 1 feature")
            return

        if st.sidebar.button("Train Model"):
            self.train_churn_model(df, selected_features, model_type)
            if model_type in self.model_results:
                st.session_state.trained_models[model_type] = self.model_results[model_type]
                st.session_state.last_trained = model_type
                st.success(f"✅ {model_type} added to global trained models.")

        # Show current (if trained)
        name_to_show = model_type if model_type in self.model_results else st.session_state.last_trained
        if name_to_show and (name_to_show in self.model_results):
            self.display_model_results(name_to_show)
        else:
            st.info("Train a model to see metrics here.")

    def train_churn_model(self, df, features, model_type):
        try:
            if len(features) == 0:
                st.error("No features selected")
                return
            if len(df) < 10:
                st.error("Dataset too small (minimum 10 rows required)")
                return
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                st.error(f"Missing features in data: {missing_features}")
                return

            X = df[features].copy()
            if X.isnull().sum().sum() > 0:
                st.warning("Found missing values, filling with median/mode")
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)

            if 'churn_risk' not in df.columns:
                st.error("churn_risk column not found in data")
                return
            y = df['churn_risk']

            class_counts = y.value_counts()
            if len(class_counts) < 2:
                st.error("Only one class found in target variable")
                return
            if class_counts.min() < 2:
                st.warning("Very few samples in minority class, results may be unreliable")

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            # model zoo
            model_configs = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=min(5, max(2, len(X_train)//10))),
                "ElasticNet": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42, max_iter=1000),
                "Ridge Regression": LogisticRegression(penalty='l2', random_state=42, max_iter=1000),
                "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42)
            }
            if model_type not in model_configs:
                st.error(f"Model type '{model_type}' not recognized")
                return
            model = model_configs[model_type]

            # scale if needed
            scaling_models = ["Logistic Regression", "SVM", "Neural Network (MLP)", "K-Nearest Neighbors", "ElasticNet", "Ridge Regression"]
            scaler = None
            X_train_processed, X_test_processed = X_train, X_test
            if model_type in scaling_models:
                try:
                    scaler = StandardScaler()
                    X_train_processed = scaler.fit_transform(X_train)
                    X_test_processed = scaler.transform(X_test)
                except Exception as e:
                    st.warning(f"Scaling failed: {str(e)}, using original data")

            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_processed)
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            else:
                if hasattr(model, 'decision_function'):
                    raw = model.decision_function(X_test_processed)
                    y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                else:
                    y_proba = y_pred.astype(float)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }
            try:
                metrics['auc'] = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
            except Exception:
                metrics['auc'] = 0.0
                st.warning("AUC calculation failed")

            self.model_results[model_type] = {
                'model': model,
                'metrics': metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'features': features,
                'scaler': scaler
            }
            st.success(f"✅ {model_type} trained successfully!")
        except Exception as e:
            st.error(f"Unexpected error in model training: {str(e)}")

    def display_model_results(self, model_type):
        try:
            if model_type not in self.model_results:
                st.error(f"No results found for {model_type}")
                return
            results = self.model_results[model_type]

            st.subheader("Model Performance")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{results['metrics']['accuracy']:.3f}")
            c2.metric("Precision", f"{results['metrics']['precision']:.3f}")
            c3.metric("Recall", f"{results['metrics']['recall']:.3f}")
            c4.metric("F1-Score", f"{results['metrics']['f1']:.3f}")
            c5.metric("AUC", f"{results['metrics']['auc']:.3f}")

            # ROC
            try:
                st.subheader("ROC Curve")
                if len(np.unique(results['y_test'])) > 1 and results['metrics']['auc'] > 0:
                    fpr, tpr, _ = roc_curve(results['y_test'], results['y_proba'])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{model_type} (AUC = {results["metrics"]["auc"]:.3f})',
                                                 line=dict(color=_palette(0))))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color=_palette(3), dash='dash')))
                    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.warning("ROC curve cannot be displayed (insufficient data or single class)")
            except Exception as e:
                st.warning(f"Could not display ROC curve: {str(e)}")

            # Importance / Coefs
            try:
                if hasattr(results['model'], 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importances = results['model'].feature_importances_
                    if len(importances) == len(results['features']):
                        feature_importance = pd.DataFrame({
                            'feature': results['features'],
                            'importance': importances
                        }).sort_values('importance', ascending=True)
                        fig_importance = px.bar(feature_importance, x='importance', y='feature',
                                                title='Feature Importance', color='feature',
                                                color_discrete_sequence=PLOTLY_PALETTE)
                        fig_importance.update_layout(showlegend=False)
                        st.plotly_chart(fig_importance, use_container_width=True)
                elif hasattr(results['model'], 'coef_'):
                    st.subheader("Feature Coefficients")
                    coefs = results['model'].coef_[0] if len(results['model'].coef_.shape) > 1 else results['model'].coef_
                    if len(coefs) == len(results['features']):
                        feature_coef = pd.DataFrame({
                            'feature': results['features'],
                            'coefficient': np.abs(coefs)
                        }).sort_values('coefficient', ascending=True)
                        fig_coef = px.bar(feature_coef, x='coefficient', y='feature',
                                          title='Feature Coefficients (Abs)', color='feature',
                                          color_discrete_sequence=PLOTLY_PALETTE)
                        fig_coef.update_layout(showlegend=False)
                        st.plotly_chart(fig_coef, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display feature importance/coefficients: {str(e)}")
        except Exception as e:
            st.error(f"Error displaying model results: {str(e)}")

    # ---------- Comparison (with quick train) ----------
    def comparison_tab(self):
        st.markdown('<div class="main-header">Model Comparison</div>', unsafe_allow_html=True)
        st.caption("ℹ️ Train additional models here or in the Churn Models tab; results persist app-wide.")

        # quick train
        with st.expander("➕ Train another model for comparison", expanded=False):
            df = self.processed_data
            if df is None:
                st.warning("Load data first.")
            else:
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    model_type = st.selectbox(
                        "Model:",
                        ["Logistic Regression", "Random Forest", "Decision Tree",
                         "Gradient Boosting", "SVM", "Neural Network (MLP)",
                         "AdaBoost", "Naive Bayes", "K-Nearest Neighbors",
                         "ElasticNet", "Ridge Regression", "Extra Trees"],
                        key="cmp_model"
                    )
                with mcol2:
                    features = st.multiselect(
                        "Features:",
                        ['age','income','spending_score','membership_years',
                         'purchase_frequency','last_purchase_amount','loyalty_score'],
                        default=['spending_score','purchase_frequency','loyalty_score'],
                        key="cmp_features"
                    )
                if st.button("Train in Comparison", key="cmp_train"):
                    if len(features) < 1:
                        st.error("Select at least one feature")
                    else:
                        self.train_churn_model(df, features, model_type)
                        if model_type in self.model_results:
                            st.session_state.trained_models[model_type] = self.model_results[model_type]
                            st.session_state.last_trained = model_type
                            st.success(f"{model_type} trained and added to comparison.")

        if len(st.session_state.trained_models) < 2:
            st.info("Train at least 2 models to compare them")
            return

        rows = []
        for mname, res in st.session_state.trained_models.items():
            rows.append({
                "Model": mname,
                "Accuracy": res['metrics'].get('accuracy', 0.0),
                "Precision": res['metrics'].get('precision', 0.0),
                "Recall": res['metrics'].get('recall', 0.0),
                "F1-Score": res['metrics'].get('f1', 0.0),
                "AUC": res['metrics'].get('auc', 0.0),
            })
        comparison_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        st.subheader("Performance Comparison (trained models)")
        st.dataframe(comparison_df.round(4), use_container_width=True)

        metric = st.selectbox("Metric to visualize:", ['Accuracy','Precision','Recall','F1-Score','AUC'], index=4)
        fig = px.bar(
            comparison_df, x='Model', y=metric,
            title=f'{metric} Comparison',
            color='Model', color_discrete_sequence=PLOTLY_PALETTE
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Thresholds ----------
    def thresholds_tab(self):
        st.markdown('<div class="main-header">Threshold Explorer</div>', unsafe_allow_html=True)
        st.caption("ℹ️ Pick any model you’ve trained; adjust the threshold and see live metrics.")
        if len(st.session_state.trained_models) == 0:
            st.warning("Please train at least one model first (any tab).")
            return

        model_names = sorted(list(st.session_state.trained_models.keys()))
        default_idx = model_names.index(st.session_state.last_trained) if st.session_state.last_trained in model_names else 0
        selected_model = st.selectbox("Select model:", model_names, index=default_idx)

        results = st.session_state.trained_models[selected_model]
        if results.get('y_proba') is None:
            st.warning("Selected model has no probability outputs; try a probabilistic model.")
            return

        threshold = st.slider("Classification Threshold:", 0.0, 1.0, 0.50, 0.01, key=f"thr_{selected_model}")
        y_pred_thr = (results['y_proba'] >= threshold).astype(int)

        acc  = accuracy_score(results['y_test'], y_pred_thr)
        prec = precision_score(results['y_test'], y_pred_thr, zero_division=0)
        rec  = recall_score(results['y_test'], y_pred_thr, zero_division=0)
        f1   = f1_score(results['y_test'], y_pred_thr, zero_division=0)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-Score", f"{f1:.3f}")

        st.subheader("Precision-Recall Tradeoff")
        try:
            precisions, recalls, thresholds_pr = precision_recall_curve(results['y_test'], results['y_proba'])
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, name='PR Curve', line=dict(color=_palette(0))))
            fig_pr.add_vline(x=rec, line_dash="dash", line_color=_palette(1), annotation_text=f"Recall @thr={threshold:.2f}")
            fig_pr.add_hline(y=prec, line_dash="dash", line_color=_palette(1), annotation_text=f"Precision @thr={threshold:.2f}")
            fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
            st.plotly_chart(fig_pr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display precision-recall curve: {str(e)}")

    # ---------- What-If ----------
    def what_if_tab(self):
        st.markdown('<div class="main-header">What-If Scenario Explorer</div>', unsafe_allow_html=True)
        st.caption("ℹ️ Select any model you’ve trained (Churn Models/Comparison). Predictions use that model’s features & scaling.")
        if len(st.session_state.trained_models) == 0:
            st.warning("Please train at least one model first (any tab).")
            return

        model_names = list(st.session_state.trained_models.keys())
        selected_model = st.selectbox("Select model for prediction:", model_names,
                                      index=model_names.index(st.session_state.last_trained) if st.session_state.last_trained in model_names else 0)

        results = st.session_state.trained_models.get(selected_model, self.model_results.get(selected_model))
        if not results:
            st.warning("Please train this model first (Churn Models or Comparison).")
            return

        st.subheader("Customer Profile Simulator")
        col1, col2 = st.columns(2)
        feature_values = {}
        with col1:
            if 'age' in results['features']:
                feature_values['age'] = st.slider("Age:", 18, 70, 35)
            if 'income' in results['features']:
                feature_values['income'] = st.slider("Income ($):", 30000, 150000, 75000, step=1000)
            if 'spending_score' in results['features']:
                feature_values['spending_score'] = st.slider("Spending Score:", 1, 100, 50)
        with col2:
            if 'membership_years' in results['features']:
                feature_values['membership_years'] = st.slider("Membership Years:", 1, 10, 5)
            if 'purchase_frequency' in results['features']:
                feature_values['purchase_frequency'] = st.slider("Purchase Frequency:", 1, 50, 25)
            if 'last_purchase_amount' in results['features']:
                feature_values['last_purchase_amount'] = st.slider("Last Purchase Amount ($):", 10, 1000, 200, step=5)
            if 'loyalty_score' in results['features']:
                feature_values['loyalty_score'] = st.slider("Loyalty Score:", 0.0, 1.0, 0.5)

        if st.button("Predict Churn Risk"):
            try:
                input_data = [[feature_values[f] for f in results['features']]]
                if results['scaler'] is not None:
                    input_data = results['scaler'].transform(input_data)

                if hasattr(results['model'], 'predict_proba'):
                    churn_proba = results['model'].predict_proba(input_data)[0, 1]
                else:
                    pred = results['model'].predict(input_data)[0]
                    churn_proba = float(pred)

                churn_pred = churn_proba >= 0.5
                st.subheader("Prediction Results")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Churn Probability", f"{churn_proba:.1%}")
                with c2:
                    risk_level = "High" if churn_proba > 0.7 else "Medium" if churn_proba > 0.3 else "Low"
                    st.metric("Risk Level", risk_level)
                with c3:
                    st.metric("Prediction", "At Risk" if churn_pred else "Retained")

                st.subheader("Recommendations")
                if churn_proba > 0.7:
                    st.error("**High Risk Customer** - Immediate intervention recommended")
                    recs = [
                        "Contact customer immediately with personalized offer",
                        "Assign dedicated account manager",
                        "Provide exclusive perks or discounts",
                        "Schedule feedback call to understand concerns"
                    ]
                elif churn_proba > 0.3:
                    st.warning("**Medium Risk Customer** - Monitor closely")
                    recs = [
                        "Send targeted marketing campaigns",
                        "Offer loyalty program benefits",
                        "Track engagement metrics closely",
                        "Consider satisfaction survey"
                    ]
                else:
                    st.success("**Low Risk Customer** - Continue standard engagement")
                    recs = [
                        "Maintain regular communication",
                        "Continue standard offers",
                        "Monitor for any changes in behavior",
                        "Consider for referral programs"
                    ]
                for i, rec in enumerate(recs, 1):
                    st.write(f"{i}. {rec}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    # ---------- Main ----------
    def run_dashboard(self):
        try:
            st.title("Customer Retention Dashboard")
            st.markdown("*Comprehensive ML-powered customer analytics and churn prediction*")

            # Load
            try:
                if not self.load_data():
                    st.error("Failed to load data. Please check your file and try again.")
                    st.stop()
            except Exception as e:
                st.error(f"Critical error loading data: {str(e)}")
                st.stop()

            # Process
            try:
                self.preprocess_data()
                if self.processed_data is None:
                    st.error("Data preprocessing failed")
                    st.stop()
            except Exception as e:
                st.error(f"Data preprocessing error: {str(e)}")
                st.stop()

            # Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Summary", "EDA", "Segmentation", "Churn Models",
                "Comparison", "Thresholds", "What-If"
            ])

            with tab1:
                try: self.summary_tab()
                except Exception as e: st.error(f"Error in Summary tab: {str(e)}")

            with tab2:
                try: self.eda_tab()
                except Exception as e: st.error(f"Error in EDA tab: {str(e)}")

            with tab3:
                try: self.segmentation_tab()
                except Exception as e: st.error(f"Error in Segmentation tab: {str(e)}")

            with tab4:
                try: self.churn_models_tab()
                except Exception as e: st.error(f"Error in Churn Models tab: {str(e)}")

            with tab5:
                try: self.comparison_tab()
                except Exception as e: st.error(f"Error in Comparison tab: {str(e)}")

            with tab6:
                try: self.thresholds_tab()
                except Exception as e: st.error(f"Error in Thresholds tab: {str(e)}")

            with tab7:
                try: self.what_if_tab()
                except Exception as e: st.error(f"Error in What-If tab: {str(e)}")

        except Exception as e:
            st.error(f"Critical dashboard error: {str(e)}")


# Run
if __name__ == "__main__":
    dashboard = CustomerRetentionDashboard()
    dashboard.run_dashboard()