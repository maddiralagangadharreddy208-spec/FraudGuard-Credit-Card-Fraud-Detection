
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="FraudGuard | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "gangadharreddy123/fraudguard-creditcard-data/"
    "resolve/main/creditcard.csv"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }
    [data-testid="stMetric"] {
        background-color: #162238;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ FraudGuard")
st.caption("Credit Card Fraud Detection using Random Forest")

# --------------------------------------------------
# LOAD DATA FROM HUGGING FACE
# --------------------------------------------------

@st.cache_data(show_spinner="Downloading transaction dataset...")
def load_data():
    data = pd.read_csv(DATASET_URL)
    data = data.drop_duplicates().reset_index(drop=True)

    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    return data


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model():
    data = load_data()

    X = data.drop(columns=["Class"])
    y = data["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
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

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            output_dict=True,
            zero_division=0,
        ),
    }

    return model, X_test, y_test, predictions, metrics


# --------------------------------------------------
# HANDLE LOADING ERRORS
# --------------------------------------------------

try:
    data = load_data()
    model, X_test, y_test, predictions, metrics = train_model()

except Exception as error:
    st.error("FraudGuard could not load the dataset or train the model.")
    st.warning(
        "Check that the Hugging Face dataset is public and the "
        "creditcard.csv file is available."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛡️ FraudGuard")
st.sidebar.caption("Security Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Fraud Detection",
        "Model Analytics",
    ],
)

st.sidebar.divider()
st.sidebar.success("Dataset loaded successfully")
st.sidebar.write(f"Transactions: {len(data):,}")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":
    st.header("📊 Dashboard")
    st.write("Overview of credit card transactions in the dataset.")

    total_transactions = len(data)
    fraud_count = int((data["Class"] == 1).sum())
    legitimate_count = int((data["Class"] == 0).sum())
    fraud_percentage = (
        fraud_count / total_transactions * 100
        if total_transactions else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Legitimate", f"{legitimate_count:,}")
    col3.metric("Fraud Cases", f"{fraud_count:,}")
    col4.metric("Fraud Rate", f"{fraud_percentage:.3f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Transaction Class Distribution")
        counts = data["Class"].value_counts().reindex(
            [0, 1], fill_value=0
        )
        counts.index = ["Legitimate", "Fraud"]
        st.bar_chart(counts)

    with right:
        st.subheader("Transaction Amounts")
        if "Amount" in data.columns:
            st.histogram if False else None
            fig, ax = plt.subplots()
            ax.hist(data["Amount"], bins=40)
            ax.set_xlabel("Transaction Amount")
            ax.set_ylabel("Number of Transactions")
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)


# --------------------------------------------------
# FRAUD DETECTION
# --------------------------------------------------

elif page == "Fraud Detection":
    st.header("🔍 Fraud Detection")
    st.write(
        "Select a transaction from the held-out test data and "
        "ask the trained model to classify it."
    )

    transaction_number = st.number_input(
        "Test transaction number",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
    )

    transaction = X_test.iloc[[int(transaction_number)]]

    with st.expander("View transaction details"):
        st.dataframe(transaction, use_container_width=True)

    if st.button("🔎 Check Transaction", type="primary"):
        result = int(model.predict(transaction)[0])
        probabilities = model.predict_proba(transaction)[0]
        fraud_probability = float(
            probabilities[list(model.classes_).index(1)]
        )

        st.metric(
            "Estimated Fraud Probability",
            f"{fraud_probability * 100:.2f}%",
        )

        if result == 1:
            st.error("⚠️ Model prediction: Potential Fraud")
        else:
            st.success("✅ Model prediction: Likely Legitimate")

        actual_class = int(y_test.iloc[int(transaction_number)])
        st.caption(
            "For this test example, the recorded dataset label is: "
            + ("Fraud" if actual_class == 1 else "Legitimate")
        )

    st.warning(
        "This is an educational project, not a real banking security "
        "system. Predictions can be wrong."
    )


# --------------------------------------------------
# MODEL ANALYTICS
# --------------------------------------------------

elif page == "Model Analytics":
    st.header("📈 Model Analytics")
    st.write("Evaluation results on the held-out test dataset.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")

    st.divider()

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(
        metrics["confusion"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns,
    ).sort_values(ascending=False).head(15)

    st.bar_chart(importance)

    st.caption(
        "Metrics are calculated from the current train/test split. "
        "High accuracy alone does not guarantee reliable fraud detection."
    )

st.divider()
st.caption("FraudGuard | Academic Machine Learning Project")
