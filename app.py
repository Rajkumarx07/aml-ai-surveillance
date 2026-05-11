%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="AI AML Surveillance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""

<style>

body {
    background-color: #071028;
}

.main {
    background: linear-gradient(
        135deg,
        #071028,
        #0b1736,
        #071028
    );
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #081120;
}

.stMetric {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
}

[data-testid="stMetricValue"] {
    color: #00E5FF;
    font-size: 34px;
}

[data-testid="stMetricLabel"] {
    color: white;
}

div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 10px;
}

.block-container {
    padding-top: 1rem;
}

</style>

""", unsafe_allow_html=True)

df = pd.read_csv("aml_transactions_dataset.csv")

original_df = df.copy()

categorical_columns = [

    "group_id",
    "mobile",
    "email",
    "ip_address",
    "address",
    "bank_account",
    "device_id",
    "occupation",
    "risk_country",
    "rapid_withdrawal"

]

encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

df["turnover_income_ratio"] = (
    df["monthly_turnover"] /
    df["declared_income"]
)

df["transaction_velocity"] = (
    df["fund_transfer_count"] *
    df["login_frequency"]
)

df["behavior_score"] = (

    df["fo_premium"] * 0.00005 +

    df["illiquid_trades"] * 4 +

    df["fund_transfer_count"] * 2 +

    df["login_frequency"] * 1.5 +

    df["shared_accounts"] * 8 +

    df["trading_pattern_score"] * 2

)

df["aml_risk_score"] = (

    df["turnover_income_ratio"] * 4 +

    df["behavior_score"] * 0.7 +

    df["transaction_amount"] * 0.00001

)

features = [

    "declared_income",
    "monthly_turnover",
    "fo_premium",
    "illiquid_trades",
    "fund_transfer_count",
    "login_frequency",
    "shared_accounts",
    "trading_pattern_score",
    "transaction_amount",
    "turnover_income_ratio",
    "transaction_velocity",
    "behavior_score",
    "aml_risk_score"

]

X = df[features]

y = df["suspicious_flag"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

model = RandomForestClassifier(

    n_estimators=100,

    max_depth=6,

    min_samples_leaf=10,

    random_state=42

)

model.fit(X_train, y_train)

predictions = model.predict_proba(X)[:,1]

noise = np.random.normal(0, 0.08, len(predictions))

predictions = predictions + noise

predictions = np.clip(predictions, 0.02, 0.98)

df["ai_suspicion_probability"] = predictions

def assign_priority(prob):

    if prob >= 0.80:
        return "Critical"

    elif prob >= 0.60:
        return "High"

    elif prob >= 0.40:
        return "Medium"

    else:
        return "Low"

df["alert_priority"] = df["ai_suspicion_probability"].apply(assign_priority)

def generate_reason(row):

    reasons = []

    if row["turnover_income_ratio"] > 18:
        reasons.append("Turnover anomaly")

    if row["fo_premium"] > 400000:
        reasons.append("Aggressive derivatives activity")

    if row["illiquid_trades"] > 5:
        reasons.append("Illiquid exposure")

    if row["fund_transfer_count"] > 12:
        reasons.append("Frequent fund transfers")

    if row["shared_accounts"] > 2:
        reasons.append("Linked entity behaviour")

    if row["login_frequency"] > 70:
        reasons.append("Abnormal login pattern")

    return ", ".join(reasons)

df["alert_reason"] = df.apply(generate_reason, axis=1)

st.sidebar.title("AML Navigation")

page = st.sidebar.radio(

    "Select Module",

    [

        "Executive Overview",
        "Alert Intelligence",
        "Relationship Intelligence",
        "Suspicious Cluster Explorer",
        "Compliance Report"

    ]

)

st.sidebar.markdown("---")

risk_filter = st.sidebar.multiselect(

    "Alert Priority",

    options=df["alert_priority"].unique(),

    default=df["alert_priority"].unique()

)

city_filter = st.sidebar.multiselect(

    "City",

    options=original_df["address"].unique(),

    default=original_df["address"].unique()

)

filtered_df = df[

    (df["alert_priority"].isin(risk_filter))

]

st.title("🛡️ AI AML Surveillance Platform")

st.markdown("""
Intelligent Alert Prioritization • Relationship Intelligence • Group Risk Detection
""")

st.markdown("---")
