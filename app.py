
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AI AML Surveillance System",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""

<style>

.main {
    background-color: #071028;
    color: white;
}

[data-testid="stMetricValue"] {
    color: #00E5FF;
    font-size: 32px;
}

[data-testid="stMetricLabel"] {
    color: white;
}

.block-container {
    padding-top: 1rem;
}

div[data-testid="stDataFrame"] {
    background-color: #101c3d;
    border-radius: 12px;
    padding: 10px;
}

</style>

""", unsafe_allow_html=True)

st.title("🛡️ AI-Powered AML Surveillance Framework")

st.markdown("""
### Intelligent AML Monitoring • Alert Prioritization • Relationship Intelligence • Group Risk Detection
""")

st.markdown("---")

df = pd.read_csv("aml_transactions_dataset.csv")

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

for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

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

    df["turnover_income_ratio"] * 5 +

    df["behavior_score"] * 0.8 +

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

model = RandomForestClassifier(

    n_estimators=120,

    max_depth=7,

    min_samples_split=10,

    min_samples_leaf=5,

    random_state=42

)

model.fit(X, y)

prediction_probabilities = model.predict_proba(X)[:,1]

df["ai_suspicion_probability"] = prediction_probabilities

def priority(prob):

    if prob >= 0.85:
        return "Critical"

    elif prob >= 0.65:
        return "High"

    elif prob >= 0.40:
        return "Medium"

    else:
        return "Low"

df["alert_priority"] = df["ai_suspicion_probability"].apply(priority)

def generate_reason(row):

    reasons = []

    if row["turnover_income_ratio"] > 20:
        reasons.append("Turnover exceeds income")

    if row["fo_premium"] > 500000:
        reasons.append("Excessive F&O activity")

    if row["illiquid_trades"] > 6:
        reasons.append("Illiquid trading")

    if row["fund_transfer_count"] > 15:
        reasons.append("Frequent fund movement")

    if row["shared_accounts"] > 2:
        reasons.append("Linked account activity")

    if row["login_frequency"] > 80:
        reasons.append("Abnormal login behaviour")

    return ", ".join(reasons)

df["alert_reason"] = df.apply(generate_reason, axis=1)

critical_alerts = len(
    df[df["alert_priority"] == "Critical"]
)

high_alerts = len(
    df[df["alert_priority"] == "High"]
)

avg_risk = round(
    df["ai_suspicion_probability"].mean() * 100,
    2
)

linked_accounts = len(
    df[df["shared_accounts"] > 2]
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Critical Alerts",
    critical_alerts
)

c2.metric(
    "High Risk Alerts",
    high_alerts
)

c3.metric(
    "Average Risk %",
    avg_risk
)

c4.metric(
    "Linked Accounts",
    linked_accounts
)

st.markdown("---")

st.subheader("📈 AML Intelligence Overview")

fig1 = px.histogram(

    df,

    x="ai_suspicion_probability",

    nbins=60,

    title="AI Suspicion Probability Distribution"

)

fig1.update_layout(
    paper_bgcolor="#071028",
    plot_bgcolor="#071028",
    font_color="white"
)

st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

st.subheader("🚨 Alert Priority Distribution")

fig2 = px.pie(

    df,

    names="alert_priority",

    hole=0.5,

    title="Alert Severity Classification"

)

fig2.update_layout(
    paper_bgcolor="#071028",
    font_color="white"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("🏦 Top Suspicious Clients")

top_clients = df.sort_values(

    by="ai_suspicion_probability",

    ascending=False

).head(20)

fig3 = px.bar(

    top_clients,

    x="client_id",

    y="ai_suspicion_probability",

    color="alert_priority",

    title="Top High-Risk Clients"

)

fig3.update_layout(
    paper_bgcolor="#071028",
    plot_bgcolor="#071028",
    font_color="white"
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

st.subheader("🧠 AI Reasoning Insights")

selected_client = st.selectbox(

    "Select Client",

    df["client_id"].unique()

)

client_data = df[
    df["client_id"] == selected_client
].iloc[0]

st.markdown(f"""
### Client Intelligence Summary

- **AI Risk Probability:** {round(client_data['ai_suspicion_probability'] * 100,2)}%
- **Alert Priority:** {client_data['alert_priority']}
- **AML Risk Score:** {round(client_data['aml_risk_score'],2)}

### Risk Indicators
{client_data['alert_reason']}
""")

st.markdown("---")

st.subheader("📊 Group Risk Intelligence")

group_risk = df.groupby("group_id").agg({

    "client_id":"count",
    "ai_suspicion_probability":"mean",
    "aml_risk_score":"mean"

}).reset_index()

group_risk.columns = [

    "group_id",
    "group_size",
    "avg_probability",
    "avg_risk_score"

]

group_risk = group_risk.sort_values(

    by="avg_probability",

    ascending=False

)

fig4 = px.scatter(

    group_risk.head(100),

    x="group_size",

    y="avg_probability",

    size="avg_risk_score",

    hover_data=["group_id"],

    title="Suspicious Group Intelligence"

)

fig4.update_layout(
    paper_bgcolor="#071028",
    plot_bgcolor="#071028",
    font_color="white"
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

st.subheader("🔍 Live AML Investigation Table")

investigation_table = df[[

    "client_id",
    "aml_risk_score",
    "ai_suspicion_probability",
    "alert_priority",
    "alert_reason"

]].sort_values(

    by="ai_suspicion_probability",

    ascending=False

)

st.dataframe(
    investigation_table.head(200),
    use_container_width=True
)
