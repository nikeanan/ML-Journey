"""
Professional, Premium UI/UX Streamlit Web Application Dashboard for ML-Journey.
Includes Custom CSV Upload Data Pipeline, 3D Cost Surface, Model Benchmarks & Neural Net.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

from ml_journey import (
    compute_cost_matrix,
    compute_gradient_matrix,
    DenseLayer,
    NeuralNetwork,
    plot_cost_surface_3d,
    plot_model_comparison_bar,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_profit_vs_threshold,
    compute_financial_profit,
    find_optimal_profit_threshold,
)



# Page Config
st.set_page_config(
    page_title="ML-Journey Pro Studio 🚀",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(129, 140, 248, 0.5);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1>⚡ ML-Journey Pro Studio</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.4rem;">
            Interactive Machine Learning Engineering Studio, Custom CSV Pipeline & Model Training
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Menu
st.sidebar.image("https://img.icons8.com/isometric/100/brain.png", width=80)
st.sidebar.title("ML Studio Modules")
module = st.sidebar.selectbox(
    "Select Lab Module",
    [
        "📈 3D Cost Surface & Gradient Descent",
        "📊 Classifier Performance Studio",
        "💳 Credit Risk Scoring & Explainable AI",
        "🔄 Customer Churn & LTV Optimizer",
        "💰 Financial ROI Campaign Optimizer",
        "🔑 Commercial API & Developer Portal",
        "📁 Custom CSV Pipeline & Live Inference",
        "🧠 Neural Network (Scratch Lab)",
    ],
)





st.sidebar.markdown("---")
st.sidebar.info("💡 **ML-Journey v1.0.0**: Created by Niketan Anand")


if module == "📈 3D Cost Surface & Gradient Descent":
    st.subheader("Linear Regression & 3D Optimization Surface")
    st.markdown("Interactively adjust model hyper-parameters and inspect 3D cost surface convergence paths in real time.")

    col1, col2 = st.columns([1, 2.2])

    with col1:
        st.markdown("### ⚙️ Parameters")
        num_samples = st.slider("Dataset Size (m)", 20, 200, 100, step=10)
        noise = st.slider("Gaussian Noise Level", 0.0, 5.0, 1.0, step=0.5)
        lr = st.select_slider("Learning Rate (α)", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
        iters = st.slider("Iterations", 10, 500, 150, step=10)

        # Synthetic Data
        np.random.seed(42)
        X = np.random.randn(num_samples, 1) * 2.0
        true_w, true_b = 3.5, 1.2
        y = (X * true_w + true_b).flatten() + np.random.randn(num_samples) * noise

        # Run Gradient Descent
        w, b = 0.0, 0.0
        history = {"params": [], "cost": []}
        for _ in range(iters):
            cost = compute_cost_matrix(X, y, np.array([w]), b)
            history["params"].append([np.array([w]), b])
            history["cost"].append(cost)
            dj_db, dj_dw = compute_gradient_matrix(X, y, np.array([w]), b)
            w -= lr * dj_dw[0]
            b -= lr * dj_db

        # KPI Metric Cards
        st.markdown("### 🎯 Model Results")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Learned Weight (w)</div><div class="metric-value">{w:.3f}</div></div>',
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Final Cost J(w,b)</div><div class="metric-value">{history["cost"][-1]:.3f}</div></div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 🌐 Interactive 3D Cost Surface J(w,b)")
        fig_3d = plot_cost_surface_3d(X, y, w_range=[-5, 12, 35], b_range=[-20, 20, 35], history=history)
        fig_3d.update_layout(template="plotly_dark", height=550)
        st.plotly_chart(fig_3d)

elif module == "📊 Classifier Performance Studio":
    st.subheader("Model Benchmarking & Evaluation Dashboard")
    st.markdown("Compare machine learning classification models on real-world datasets.")

    benchmark_data = {
        "Gradient Boosting": {"ROC-AUC": 0.925, "F1 Score": 0.881, "Precision": 0.865, "Recall": 0.898},
        "Random Forest": {"ROC-AUC": 0.912, "F1 Score": 0.865, "Precision": 0.842, "Recall": 0.890},
        "Support Vector Machine": {"ROC-AUC": 0.884, "F1 Score": 0.829, "Precision": 0.830, "Recall": 0.828},
        "Logistic Regression": {"ROC-AUC": 0.835, "F1 Score": 0.771, "Precision": 0.790, "Recall": 0.753},
    }

    st.success("🏆 **Best Model**: **Gradient Boosting Classifier** achieves the highest discriminatory power with an **ROC-AUC of 0.925**.")

    tab1, tab2 = st.tabs(["📊 Metric Charts", "📋 Benchmark Leaderboard"])

    with tab1:
        fig_bar = plot_model_comparison_bar(benchmark_data)
        fig_bar.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_bar)

    with tab2:
        df_metrics = pd.DataFrame(benchmark_data).T
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color="#1e3a8a"),
        )

elif module == "💳 Credit Risk Scoring & Explainable AI":
    st.subheader("Credit Card Default Risk Scoring & Explainable AI 💳")
    st.markdown(
        "Translate default probabilities into standard **FICO-style Credit Risk Scores (300 - 850)** and analyze risk factors."
    )

    c1, c2 = st.columns([1, 1.8])

    with c1:
        st.markdown("### 👤 Applicant Profile Simulation")
        limit_bal = st.slider("Credit Limit ($)", 10000, 500000, 50000, step=10000)
        pay_status = st.selectbox("Past Payment Delay Status", ["On Time (0)", "1 Month Delay (1)", "2 Months Delay (2)", "3+ Months Delay (3)"])
        bill_amt = st.number_input("Current Bill Amount ($)", value=25000)
        pay_amt = st.number_input("Last Payment Amount ($)", value=1500)

        # Map delay score
        delay_val = 0 if "On Time" in pay_status else int(pay_status.split("(")[1][0])
        default_prob_est = np.clip(0.05 + (delay_val * 0.22) + ((bill_amt - pay_amt) / (limit_bal + 1e-5)) * 0.15, 0.01, 0.95)

        from ml_journey import probability_to_credit_score, get_credit_risk_tier, plot_credit_score_gauge
        credit_score = probability_to_credit_score(default_prob_est)
        tier_label, color_code = get_credit_risk_tier(credit_score)

        st.markdown("### 🎯 Credit Scorecard")
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Calculated Credit Score</div><div class="metric-value" style="color:{color_code};">{credit_score}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Risk Tier Category</div><div class="metric-value" style="font-size:1.2rem; color:{color_code};">{tier_label}</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        fig_gauge = plot_credit_score_gauge(credit_score, tier_label)
        st.plotly_chart(fig_gauge)

        st.subheader("Top Predictive Credit Risk Drivers")
        mock_features = ["PAY_0 (Recent Delay)", "LIMIT_BAL (Credit Limit)", "BILL_AMT1 (Current Debt)", "PAY_AMT1 (Recent Payment)", "AGE"]
        mock_importances = np.array([0.45, 0.22, 0.15, 0.12, 0.06])
        fig_feat = plot_feature_importance(mock_features, mock_importances, top_n=5)
        fig_feat.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_feat)

elif module == "🔄 Customer Churn & LTV Optimizer":
    st.subheader("Customer Churn Risk & LTV Incentive Optimizer 🔄")
    st.markdown(
        "Predict customer churn risk, estimate Customer Lifetime Value (LTV), and calculate maximum profitable retention discount offers."
    )

    c1, c2 = st.columns([1, 1.8])

    with c1:
        st.markdown("### 👤 Subscriber Profile")
        monthly_spend = st.number_input("Monthly Bill Amount ($)", value=85.0, step=5.0)
        tenure = st.slider("Contract Tenure (Months)", 1, 72, 18)
        churn_risk_slider = st.slider("Predicted Churn Risk Probability", 0.05, 0.95, 0.72, step=0.01)

        from ml_journey import calculate_customer_ltv, calculate_max_retention_discount
        ltv_val = calculate_customer_ltv(monthly_spend, tenure)
        ret_res = calculate_max_retention_discount(churn_risk_slider, ltv_val)

        st.markdown("### 🎯 Retention Decision & Offer")
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Estimated Lifetime Value (LTV)</div><div class="metric-value">${ltv_val:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Max Allowed Retention Offer</div><div class="metric-value" style="color:#c084fc;">${ret_res["Max Retention Discount ($)"]:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.info(f"Priority Status: **{ret_res['Retention Priority']}**")

    with c2:
        st.markdown("### 📊 Portfolio Subscriber Churn Risk Matrix")
        np.random.seed(42)
        sim_data = []
        for i in range(15):
            b = np.random.uniform(40, 150)
            t = np.random.randint(6, 48)
            p = np.random.uniform(0.1, 0.9)
            l = calculate_customer_ltv(b, t)
            r = calculate_max_retention_discount(p, l)
            sim_data.append({
                "Subscriber": f"Sub_{i+101}",
                "Monthly Bill ($)": round(b, 2),
                "Tenure (Mo)": t,
                "Churn Prob": round(p, 2),
                "LTV ($)": round(l, 2),
                "Max Offer ($)": round(r["Max Retention Discount ($)"], 2),
                "Priority": r["Retention Priority"]
            })
        df_sim = pd.DataFrame(sim_data)
        st.dataframe(df_sim, use_container_width=True)

elif module == "💰 Financial ROI Campaign Optimizer":
    st.subheader("Financial ROI & Decision Threshold Campaign Optimizer 💰")
    st.markdown(
        "Optimize marketing campaign profitability by shifting classification decision thresholds based on call costs vs conversion value."
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("### 💵 Cost & Revenue Config")
        cost_per_call = st.number_input("Cost per Call ($)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        value_per_conv = st.number_input("Revenue per Conversion ($)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)

        # Simulation on synthetic campaign probabilities
        np.random.seed(42)
        y_sim = np.random.binomial(1, 0.12, 5000)
        y_prob_sim = np.clip(y_sim * 0.7 + np.random.beta(2, 5, 5000) * 0.4, 0, 1)

        from ml_journey import find_optimal_profit_threshold, compute_financial_profit
        from ml_journey import plot_profit_vs_threshold

        best_th, max_p, curve = find_optimal_profit_threshold(
            y_sim, y_prob_sim, cost_per_call=cost_per_call, value_per_conversion=value_per_conv
        )
        std_p = compute_financial_profit(
            y_sim, y_prob_sim, threshold=0.50, cost_per_call=cost_per_call, value_per_conversion=value_per_conv
        )["Net Profit"]

        st.markdown("### 📊 Financial ROI Impact")
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Optimal Decision Cutoff</div><div class="metric-value">{best_th:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Max Net Profit</div><div class="metric-value">${max_p:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Profit Lift vs Standard (0.50)</div><div class="metric-value">+${max_p - std_p:,.0f}</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        fig_p = plot_profit_vs_threshold(curve, best_th, max_p)
        fig_p.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_p)

elif module == "🔑 Commercial API & Developer Portal":
    st.subheader("Commercial API Developer Portal & Monetization 🔑")
    st.markdown(
        "Integrate production REST API endpoints into external banking systems, mobile apps, or frontend web platforms."
    )

    col_api1, col_api2 = st.columns([1, 1.8])

    with col_api1:
        st.markdown("### 🎫 API Key Management")
        st.info("🔒 **Active Sandbox API Key**: `sk_demo_free_123456789`")
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">API Billing Tier</div><div class="metric-value">Pro SaaS ($299/mo)</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Monthly API Quota</div><div class="metric-value">50,000 / 100,000</div></div>',
            unsafe_allow_html=True,
        )
        st.success("✅ FastAPI Server Status: **Operational (`/v1/credit-score`)**")

    with col_api2:
        st.markdown("### 💻 Integration Code Snippet (Python `requests`)")
        st.code(
            """import requests

url = "http://localhost:8000/v1/credit-score"
headers = {"X-API-Key": "sk_demo_free_123456789"}
payload = {
    "credit_limit": 50000.0,
    "payment_delay_status": 1,
    "current_bill_amount": 25000.0,
    "last_payment_amount": 1500.0
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
# Output: {'credit_score': 642, 'risk_tier': 'High Risk ⚠️', 'underwriting_decision': 'REJECTED'}
""",
            language="python",
        )

        st.markdown("### 🌐 Test cURL Command")
        st.code(
            """curl -X POST "http://localhost:8000/v1/credit-score" \\
     -H "Content-Type: application/json" \\
     -H "X-API-Key: sk_demo_free_123456789" \\
     -d '{"credit_limit":50000,"payment_delay_status":1,"current_bill_amount":25000,"last_payment_amount":1500}'""",
            language="bash",
        )






elif module == "📁 Custom CSV Pipeline & Live Inference":
    st.subheader("Custom CSV Data Pipeline & Live Model Inference")
    st.markdown("Upload any CSV dataset, automatically preprocess features, train a classifier live, and inspect metrics.")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        delimiter = st.radio("CSV Delimiter", [",", ";", "\\t"], horizontal=True)
        df_upload = pd.read_csv(uploaded_file, sep=delimiter)
        st.dataframe(df_upload.head(5))

        col_tgt, col_algo = st.columns(2)
        with col_tgt:
            target_col = st.selectbox("Select Target Column (y)", df_upload.columns)
        with col_algo:
            model_type = st.selectbox("Select Classification Algorithm", ["Random Forest", "Logistic Regression", "Gradient Boosting"])

        if st.button("🚀 Train Model on Custom Dataset"):
            X_cust = df_upload.drop(columns=[target_col])
            y_cust = df_upload[target_col].copy()

            # Binary encoding if text target
            if y_cust.dtype == object or str(y_cust.dtype) == "string":
                labels = y_cust.unique()
                if len(labels) >= 2:
                    y_cust = y_cust.map({labels[0]: 0, labels[1]: 1})

            # Filter high cardinality text ID columns to prevent memory overflow
            high_cardinality = [c for c in X_cust.columns if X_cust[c].dtype in [object, 'string'] and X_cust[c].nunique() > 50]
            if high_cardinality:
                X_cust = X_cust.drop(columns=high_cardinality)
                st.info(f"Dropped high-cardinality ID columns to prevent memory overload: {high_cardinality}")

            # Identify column types cleanly across Pandas 2 & 3
            num_cols = X_cust.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
            cat_cols = X_cust.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

            transformers = []
            if num_cols:
                transformers.append(('num', StandardScaler(), num_cols))
            if cat_cols:
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore'), cat_cols))

            if not transformers:
                st.error("No valid numeric or categorical features found after filtering.")
            else:
                preprocessor = ColumnTransformer(transformers=transformers)

                if model_type == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, class_weight='balanced', n_jobs=1)
                elif model_type == "Gradient Boosting":
                    clf = GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42)
                else:
                    clf = LogisticRegression(random_state=42, max_iter=1000)

                pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])

                X_tr, X_te, y_tr, y_te = train_test_split(X_cust, y_cust, test_size=0.2, random_state=42)
                
                with st.spinner("Training Model..."):
                    pipeline.fit(X_tr, y_tr)

                y_pred = pipeline.predict(X_te)
                
                # Check for multiclass vs binary target
                unique_classes = np.unique(y_te)
                is_binary = len(unique_classes) == 2

                auc = 0.0
                try:
                    if is_binary and hasattr(pipeline, "predict_proba"):
                        y_prob = pipeline.predict_proba(X_te)[:, 1]
                        auc = roc_auc_score(y_te, y_prob)
                        st.success(f"Model Training Complete! **ROC-AUC Score: {auc:.4f}**")
                    elif not is_binary and hasattr(pipeline, "predict_proba"):
                        y_prob = pipeline.predict_proba(X_te)
                        if y_prob.shape[1] == len(unique_classes):
                            auc = roc_auc_score(y_te, y_prob, multi_class="ovr")
                            st.success(f"Multi-Class Model Training Complete! **ROC-AUC (OvR): {auc:.4f}**")
                        else:
                            st.success("Multi-Class Model Training Complete!")
                    else:
                        st.success("Model Training Complete!")
                except Exception as e:
                    st.warning(f"Model trained successfully. ROC-AUC calculation skipped: {e}")

                k1, k2 = st.columns(2)
                with k1:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">ROC-AUC Score</div><div class="metric-value">{auc:.4f}</div></div>', unsafe_allow_html=True)
                with k2:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Test Examples</div><div class="metric-value">{len(y_te)}</div></div>', unsafe_allow_html=True)

                tab_diag1, tab_diag2, tab_diag3 = st.tabs(["📉 ROC Curve", "🧩 Confusion Matrix", "📋 Text Report"])

                with tab_diag1:
                    if is_binary and hasattr(pipeline, "predict_proba"):
                        y_prob = pipeline.predict_proba(X_te)[:, 1]
                        fpr, tpr, _ = roc_curve(y_te, y_prob)
                        fig_roc = plot_roc_curve(fpr, tpr, auc)
                        st.plotly_chart(fig_roc)
                    else:
                        st.info("ROC Curve available for binary classification targets.")

                with tab_diag2:
                    cm = confusion_matrix(y_te, y_pred)
                    fig_cm = plot_confusion_matrix(cm, labels=[str(c) for c in unique_classes])
                    st.plotly_chart(fig_cm)

                with tab_diag3:
                    st.code(classification_report(y_te, y_pred, zero_division=0))

                # Download Trained Model Artifact
                import io
                import joblib
                buffer = io.BytesIO()
                joblib.dump(pipeline, buffer)
                buffer.seek(0)
                st.download_button(
                    label="💾 Download Trained Model (.joblib)",
                    data=buffer,
                    file_name="trained_ml_pipeline.joblib",
                    mime="application/octet-stream",
                )






elif module == "🧠 Neural Network (Scratch Lab)":
    st.subheader("Multi-Layer Perceptron (MLP) built from Scratch")
    st.markdown("Train a Multi-Layer Neural Network using vectorized NumPy matrix computations.")

    col_cfg, col_vis = st.columns([1, 2])

    with col_cfg:
        st.markdown("### 🛠️ Architecture Config")
        hidden_units = st.slider("Hidden Layer Neurons", 2, 32, 8)
        epochs = st.slider("Training Epochs", 100, 3000, 1000, step=100)
        lr = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.5], value=0.1)

        train_btn = st.button("🚀 Train Neural Network", type="primary")

    with col_vis:
        if train_btn:
            X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
            y_xor = np.array([0, 1, 1, 0], dtype=float)

            nn = NeuralNetwork()
            nn.add(DenseLayer(2, hidden_units, activation="relu"))
            nn.add(DenseLayer(hidden_units, 1, activation="sigmoid"))

            with st.spinner("Backpropagating gradients..."):
                cost_history = nn.fit(X_xor, y_xor, epochs=epochs, learning_rate=lr)

            fig_loss = px.line(
                x=list(range(len(cost_history))),
                y=cost_history,
                title="Cross-Entropy Loss Convergence Curve",
                labels={"x": "Epoch", "y": "Loss J(w,b)"},
                template="plotly_dark",
            )
            fig_loss.update_traces(line_color="#38bdf8", line_width=3)
            st.plotly_chart(fig_loss)

            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Final Binary Cross-Entropy Loss</div><div class="metric-value">{cost_history[-1]:.6f}</div></div>',
                unsafe_allow_html=True,
            )
