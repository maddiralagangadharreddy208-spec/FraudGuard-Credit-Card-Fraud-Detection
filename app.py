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

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="FraudGuard | Security Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PROFESSIONAL DESIGN ----------------
st.markdown(
    """
    <style>
    :root {
        --background: #071326;
        --panel: #102442;
        --panel-light: #16345b;
        --border: #294b76;
        --text: #f2f6ff;
        --muted: #b7c9e4;
        --cyan: #32d4e7;
        --blue: #398bff;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, #173e73 0%, transparent 30%),
            linear-gradient(135deg, #071326 0%, #0a1931 55%, #10264a 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071326 0%, #102747 100%);
        border-right: 1px solid var(--border);
    }

    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    .hero {
        background: linear-gradient(115deg, #087cf0 0%, #276de1 48%, #7559e8 100%);
        padding: 30px 32px;
        border-radius: 22px;
        margin: 4px 0 24px 0;
        border: 1px solid #79c5ff;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.22);
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }

    .hero p {
        margin: 10px 0 0 0;
        color: #e6f1ff !important;
        font-size: 15px;
    }

    .eyebrow {
        color: #bfefff !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(22, 54, 94, 0.98), rgba(28, 75, 130, 0.95));
        border: 1px solid #3f75b4;
        border-radius: 17px;
        padding: 19px 20px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.13);
    }

    [data-testid="stMetricLabel"] p {
        color: #c9ddfa !important;
        font-size: 13px;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #08bfcf, #1686ff);
        color: white;
        border: 0;
        border-radius: 11px;
        padding: 12px 18px;
        font-weight: 750;
        min-height: 44px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        border: 1px solid #d8f5ff;
        color: white;
        filter: brightness(1.08);
    }

    .panel {
        background: linear-gradient(145deg, rgba(17, 42, 73, 0.96), rgba(19, 51, 88, 0.96));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 22px;
        margin: 8px 0;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 12px;
        font-weight: 800;
        background: rgba(50, 212, 231, 0.12);
        border: 1px solid rgba(50, 212, 231, 0.55);
        color: #aaf6ff;
    }

    .small-note {
        color: #b7c9e4 !important;
        font-size: 12px;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #2d4f79;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #29476e;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    data = data.drop_duplicates().reset_index(drop=True)
    return data


# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight={0: 1, 1: 3},
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "matrix": confusion_matrix(y_test, predictions),
        "report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "predictions": predictions,
    }

    return data, X_test, y_test, model, metrics


try:
    data, X_test, y_test, model, metrics = train_model()
except Exception as error:
    st.error("Unable to load the dataset or train the model.")
    st.write("Check that creditcard.csv is in the same folder as app.py.")
    st.exception(error)
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 💳 FraudGuard")
    st.caption("CREDIT CARD SECURITY ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Fraud Detection", "Model Analytics"],
    )

    st.divider()
    st.metric("Dataset Transactions", f"{len(data):,}")
    st.metric("Fraud Cases", f"{int(data['Class'].sum()):,}")
    st.markdown(
        '<p class="small-note">Academic machine-learning demonstration</p>',
        unsafe_allow_html=True,
    )

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">SECURITY OPERATIONS CENTER</div>
            <h1>💳 FraudGuard</h1>
            <p>Credit Card Fraud Detection · Intelligent transaction analysis powered by a Random Forest model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(data)
    fraud = int(data["Class"].sum())
    legitimate = total - fraud
    fraud_rate = fraud / total * 100

    st.markdown("### Portfolio overview")
    a, b, c, d = st.columns(4)
    a.metric("Total Transactions", f"{total:,}")
    b.metric("Legitimate", f"{legitimate:,}")
    c.metric("Fraud Cases", f"{fraud:,}")
    d.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.markdown(
        '<span class="status-pill">● DATASET LOADED</span> '
        '<span class="status-pill">● RANDOM FOREST MODEL READY</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("📊 Transaction distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#102442")
        ax.set_facecolor("#102442")
        bars = ax.bar(
            ["Legitimate", "Fraud"],
            [legitimate, fraud],
            color=["#32b9e8", "#ff6685"],
            width=0.55,
        )
        ax.set_ylabel("Number of transactions", color="#dceaff")
        ax.tick_params(colors="#dceaff")
        ax.spines["bottom"].set_color("#52739b")
        ax.spines["left"].set_color("#52739b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18, color="#9ebbe0")
        ax.set_axisbelow(True)
        ax.bar_label(bars, fmt="%.0f", color="#ffffff", padding=4, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.subheader("🍩 Dataset composition")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#102442")
        wedges, texts, autotexts = ax.pie(
            [legitimate, fraud],
            labels=["Legitimate", "Fraud"],
            autopct="%1.2f%%",
            startangle=90,
            colors=["#32b9e8", "#ff6685"],
            wedgeprops={"width": 0.42, "edgecolor": "#102442", "linewidth": 2},
            textprops={"color": "#f2f6ff", "fontsize": 10},
        )
        for autotext in autotexts:
            autotext.set_color("#ffffff")
            autotext.set_fontweight("bold")
        ax.set_facecolor("#102442")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.subheader("🔎 Dataset preview")
    st.caption("The first 10 rows after duplicate records have been removed.")
    st.dataframe(data.head(10), width="stretch", hide_index=True)

    st.info(
        "Fraud transactions are rare in this dataset. Accuracy alone can be misleading, "
        "so FraudGuard also reports fraud precision, recall and F1-score."
    )

# ---------------- FRAUD DETECTION ----------------
elif page == "Fraud Detection":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">TRANSACTION INVESTIGATION</div>
            <h1>🔍 Fraud Detection</h1>
            <p>Choose a held-out test transaction and inspect the model's prediction and estimated risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fraud_positions = np.flatnonzero(y_test.to_numpy() == 1)
    legitimate_positions = np.flatnonzero(y_test.to_numpy() == 0)

    choice = st.radio(
        "Choose a demonstration scenario",
        [
            "Test a legitimate transaction",
            "Test a real fraud transaction",
            "Choose test transaction index",
        ],
        horizontal=True,
    )

    if choice == "Test a legitimate transaction":
        selected_index = int(legitimate_positions[0])
        st.info("This sample's actual dataset label is legitimate.")
    elif choice == "Test a real fraud transaction":
        selected_index = int(fraud_positions[0])
        st.warning(
            "This sample's actual dataset label is fraud. The model may or may not "
            "classify it correctly; this is useful for demonstrating model evaluation."
        )
    else:
        selected_index = int(
            st.number_input(
                "Test transaction index",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
                step=1,
            )
        )

    if st.button("🔎 Analyze transaction", use_container_width=True):
        transaction = X_test.iloc[[selected_index]]
        prediction = int(model.predict(transaction)[0])
        probability = float(model.predict_proba(transaction)[0][1])
        actual = int(y_test.iloc[selected_index])

        if probability < 0.20:
            risk_label = "LOW"
            risk_message = "The model estimates a low fraud probability for this sample."
        elif probability < 0.60:
            risk_label = "MEDIUM"
            risk_message = "The model estimates a medium fraud probability; further review may be appropriate."
        else:
            risk_label = "HIGH"
            risk_message = "The model estimates a high fraud probability; review this sample carefully."

        st.divider()
        st.subheader("Analysis result")
        result_col, probability_col, risk_col = st.columns(3)

        with result_col:
            if prediction == 1:
                st.error("⚠️ Model prediction: POSSIBLE FRAUD")
            else:
                st.success("✅ Model prediction: LIKELY LEGITIMATE")

        with probability_col:
            st.metric("Estimated Fraud Probability", f"{probability:.2%}")

        with risk_col:
            st.metric("Model Risk Level", risk_label)

        st.progress(float(np.clip(probability, 0.0, 1.0)))
        st.caption(risk_message)

        st.write(
            "**Actual dataset label:**",
            "Fraud" if actual == 1 else "Legitimate",
        )

        if prediction == actual:
            st.success("Correct prediction: the model matches the actual dataset label.")
        else:
            st.warning("Incorrect prediction: the model does not match the actual dataset label.")

        st.subheader("Transaction features")
        st.dataframe(transaction, width="stretch")

        st.caption(
            "Academic demonstration only. The risk level is based on the model's predicted "
            "probability and is not a real banking decision or a guarantee of transaction safety."
        )

# ---------------- MODEL ANALYTICS ----------------
elif page == "Model Analytics":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">MODEL PERFORMANCE</div>
            <h1>📊 Model Analytics</h1>
            <p>Evaluation results from the held-out test dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    b.metric("Fraud Precision", f"{metrics['precision']:.2%}")
    c.metric("Fraud Recall", f"{metrics['recall']:.2%}")
    d.metric("Fraud F1-Score", f"{metrics['f1']:.2%}")

    st.divider()
    st.subheader("Confusion matrix")
    st.caption("Rows represent actual labels; columns represent predicted labels.")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#102442")
    sns.heatmap(
        metrics["matrix"],
        annot=True,
        fmt=",d",
        cmap="mako",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=1,
        linecolor="#102442",
        ax=ax,
    )
    ax.set_xlabel("Predicted label", color="#dceaff", labelpad=10)
    ax.set_ylabel("Actual label", color="#dceaff", labelpad=10)
    ax.tick_params(colors="#dceaff")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Classification report")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df.round(4), width="stretch")

    st.subheader("Feature importance")
    importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#102442")
    ax.set_facecolor("#102442")
    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color="#32b9e8",
    )
    ax.set_xlabel("Relative importance", color="#dceaff")
    ax.tick_params(colors="#dceaff")
    ax.spines["bottom"].set_color("#52739b")
    ax.spines["left"].set_color("#52739b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18, color="#9ebbe0")
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Model information")
    st.markdown(
        """
        <div class="panel">
            <p><b>Algorithm:</b> Random Forest Classifier</p>
            <p><b>Number of trees:</b> 100</p>
            <p><b>Training/testing split:</b> 80% / 20%</p>
            <p><b>Class balancing:</b> Enabled using class weights</p>
            <p><b>Fraud label:</b> 1</p>
            <p><b>Legitimate label:</b> 0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Accuracy alone is not enough for fraud detection. Fraud recall shows how many actual "
        "fraud cases were found, while precision shows how many predicted fraud cases were truly fraud."
    )

st.markdown("---")
st.markdown(
    "<p class='small-note' style='text-align:center;'>"
    "FraudGuard · Credit Card Fraud Detection · Academic Machine Learning Project"
    "</p>",
    unsafe_allow_html=True,
)
