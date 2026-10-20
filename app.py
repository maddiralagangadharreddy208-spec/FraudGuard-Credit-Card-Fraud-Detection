
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudGuard | Security Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07152f 0%, #102b59 100%);
        color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background: #08172f;
    }

    .hero {
        background: linear-gradient(110deg, #06b6d4, #2563eb, #6246ea);
        padding: 28px;
        border-radius: 20px;
        color: white;
        margin-bottom: 22px;
    }

    .metric-card {
        background: linear-gradient(140deg, #087bd5, #0754b8);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 20px;
        min-height: 120px;
    }

    .metric-label {
        font-size: 14px;
        color: #dbeafe;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
        color: white;
    }

    .notice {
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid #0891b2;
        padding: 14px;
        border-radius: 12px;
        margin: 12px 0;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REUSABLE COMPONENTS
# ============================================================

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


# ============================================================
# LOAD REAL DATASET
# ============================================================

@st.cache_data(show_spinner="Loading and preparing transaction data...")
def load_data():
    dataset_path = Path(__file__).resolve().parent / "creditcard.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Place the dataset "
            "in the same folder as app.py."
        )

    data = pd.read_csv(dataset_path)

    required_columns = {"Time", "Amount", "Class"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "The CSV does not contain the expected credit card "
            "dataset columns, including Time, Amount, and Class."
        )

    if data["Class"].isna().any():
        data = data.dropna(subset=["Class"])

    data = data.drop_duplicates().reset_index(drop=True)

    if not set(data["Class"].unique()).issubset({0, 1}):
        raise ValueError("The Class column must contain labels 0 and 1.")

    if data["Class"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both legitimate and fraud examples."
        )

    return data


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

@st.cache_resource(show_spinner="Training the Random Forest model...")
def train_model():
    data = load_data()

    X = data.drop(columns=["Class"])
    y = data["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ),
        "report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            output_dict=True,
            zero_division=0,
        ),
    }

    return (
        data,
        model,
        X_test,
        y_test,
        y_pred,
        y_prob,
        metrics,
    )


# ============================================================
# START APPLICATION
# ============================================================

try:
    (
        data,
        model,
        X_test,
        y_test,
        y_pred,
        y_prob,
        metrics,
    ) = train_model()

except Exception as error:
    st.error("FraudGuard could not load the dataset or train the model.")
    st.code(str(error))
    st.info(
        "Check that creditcard.csv is in the same folder as app.py "
        "and that the required Python packages are installed."
    )
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("# 💳 FraudGuard")
    st.caption("CREDIT CARD SECURITY ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Fraud Detection",
            "Model Analytics",
        ],
    )

    st.divider()

    st.markdown("### Dataset Summary")
    st.metric("Dataset Transactions", f"{len(data):,}")
    st.metric("Fraud Cases", f"{int(data['Class'].sum()):,}")

    st.caption("Academic machine-learning demonstration")


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <p><b>SECURITY OPERATIONS CENTER</b></p>
            <h1>💳 FraudGuard</h1>
            <p>
                Credit Card Fraud Detection · Intelligent transaction
                analysis powered by a Random Forest model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title(
        "Portfolio Overview",
        "Summary of the dataset used for model development.",
    )

    total_transactions = len(data)
    fraud_cases = int(data["Class"].sum())
    legitimate_cases = total_transactions - fraud_cases
    fraud_rate = (
        fraud_cases / total_transactions * 100
        if total_transactions
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Transactions", f"{total_transactions:,}")

    with col2:
        metric_card("Legitimate", f"{legitimate_cases:,}")

    with col3:
        metric_card("Fraud Cases", f"{fraud_cases:,}")

    with col4:
        metric_card("Fraud Rate", f"{fraud_rate:.3f}%")

    st.markdown("")
    st.markdown(
        """
        <div class="notice">
            <b>● Dataset loaded</b> &nbsp; | &nbsp;
            <b>● Random Forest model ready</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Transaction Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = plt.subplots(figsize=(7, 4))

        labels = ["Legitimate", "Fraud"]
        values = [legitimate_cases, fraud_cases]

        ax.bar(labels, values)
        ax.set_title("Transaction Class Counts")
        ax.set_ylabel("Number of Transactions")
        ax.grid(axis="y", alpha=0.2)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with chart_col2:
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.pie(
            values,
            labels=labels,
            autopct="%1.2f%%",
            startangle=90,
        )
        ax.set_title("Legitimate vs Fraud")

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    section_title("Recent Dataset Records")

    st.dataframe(
        data.tail(10),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# PAGE 2: FRAUD DETECTION
# ============================================================

elif page == "Fraud Detection":
    st.markdown(
        """
        <div class="hero">
            <h1>🔍 Fraud Detection Center</h1>
            <p>
                Select a transaction from the held-out test dataset
                and inspect the model's prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "This is an academic demonstration using historical dataset "
        "records. It is not a live payment gateway."
    )

    section_title("Select a Test Transaction")

    transaction_number = st.number_input(
        "Test transaction number",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
    )

    transaction = X_test.iloc[[int(transaction_number)]]
    actual_label = int(y_test.iloc[int(transaction_number)])

    predicted_label = int(model.predict(transaction)[0])
    fraud_probability = float(
        model.predict_proba(transaction)[0][
            list(model.classes_).index(1)
        ]
    )

    risk_level = (
        "HIGH"
        if fraud_probability >= 0.70
        else "MEDIUM"
        if fraud_probability >= 0.30
        else "LOW"
    )

    st.markdown("---")
    section_title("Analysis Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        if predicted_label == 1:
            metric_card("Model Prediction", "POSSIBLE FRAUD")
        else:
            metric_card("Model Prediction", "LEGITIMATE")

    with result_col2:
        metric_card(
            "Estimated Fraud Probability",
            f"{fraud_probability * 100:.2f}%",
        )

    with result_col3:
        metric_card("Model Risk Level", risk_level)

    st.progress(fraud_probability)

    st.caption(
        "The displayed probability is the model's estimated score, "
        "not a guarantee that a transaction is fraudulent."
    )

    actual_text = "Fraud" if actual_label == 1 else "Legitimate"
    predicted_text = "Fraud" if predicted_label == 1 else "Legitimate"

    st.write(f"**Actual dataset label:** {actual_text}")
    st.write(f"**Model prediction:** {predicted_text}")

    if predicted_label == actual_label:
        st.success("The prediction matches the actual dataset label.")
    else:
        st.error("The prediction does not match the actual dataset label.")

    section_title("Transaction Features")

    st.dataframe(
        transaction.reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# PAGE 3: MODEL ANALYTICS
# ============================================================

elif page == "Model Analytics":
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Model Analytics</h1>
            <p>
                Evaluation results from the held-out test dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Accuracy", f"{metrics['accuracy']:.2%}")

    with col2:
        metric_card("Fraud Precision", f"{metrics['precision']:.2%}")

    with col3:
        metric_card("Fraud Recall", f"{metrics['recall']:.2%}")

    with col4:
        metric_card("Fraud F1-Score", f"{metrics['f1']:.2%}")

    st.markdown("---")
    section_title("Confusion Matrix")

    st.caption(
        "Rows represent actual labels; columns represent predicted labels."
    )

    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="winter",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("---")
    section_title("Classification Report")

    report_df = pd.DataFrame(metrics["report"]).transpose()

    st.dataframe(
        report_df.round(4),
        width="stretch",
    )

    st.markdown("---")
    section_title("Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    top_features = importance_df.head(10).sort_values(
        "Importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    ax.set_title("Top 10 Important Features")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.2)

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    with st.expander("View all feature importances"):
        st.dataframe(
            importance_df.reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )

    st.markdown("---")
    section_title("Model Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(
            """
            **Algorithm:** Random Forest Classifier

            **Number of trees:** 100

            **Training/testing split:** 80% / 20%

            **Class balancing:** Enabled using class weights
            """
        )

    with info_col2:
        st.markdown(
            """
            **Fraud label:** 1

            **Legitimate label:** 0

            **Evaluation:** Held-out test dataset
            """
        )

    st.info(
        "Accuracy alone is not enough for fraud detection. "
        "Recall measures how many actual fraud cases were found, "
        "while precision measures how many predicted fraud cases "
        "were truly fraud."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <p style="text-align:center; color:#cbd5e1;">
        FraudGuard · Credit Card Fraud Detection ·
        Academic Machine Learning Project
    </p>
    """,
    unsafe_allow_html=True,
)
