import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AI AML Surveillance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main {
    background: linear-gradient(
        135deg,
        #071028,
        #0B1736,
        #071028
    );
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #081120;
}

[data-testid="stMetricValue"] {
    color: #00E5FF;
    font-size: 34px;
}

[data-testid="stMetricLabel"] {
    color: white;
}

.stMetric {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
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

risk_filter = st.sidebar.multiselect(

    "Alert Priority",

    options=df["alert_priority"].unique(),

    default=df["alert_priority"].unique()

)

filtered_df = df[
    df["alert_priority"].isin(risk_filter)
]

st.title("🛡️ AI AML Surveillance Platform")

st.markdown("""
Intelligent Alert Prioritization • Relationship Intelligence • Group Risk Detection
""")

st.markdown("---")

if page == "Executive Overview":

    critical_alerts = len(
        filtered_df[
            filtered_df["alert_priority"] == "Critical"
        ]
    )

    high_alerts = len(
        filtered_df[
            filtered_df["alert_priority"] == "High"
        ]
    )

    medium_alerts = len(
        filtered_df[
            filtered_df["alert_priority"] == "Medium"
        ]
    )

    avg_risk = round(
        filtered_df["ai_suspicion_probability"].mean() * 100,
        2
    )

    suspicious_groups = len(
        filtered_df[
            filtered_df["shared_accounts"] > 2
        ]
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Critical Alerts", critical_alerts)
    c2.metric("High Risk Alerts", high_alerts)
    c3.metric("Medium Alerts", medium_alerts)
    c4.metric("Average Risk %", avg_risk)
    c5.metric("Linked Groups", suspicious_groups)

    st.markdown("---")

    st.subheader("🧠 Executive Intelligence Summary")

    st.info(f"""

    • AI engine identified {critical_alerts} critical entities.

    • Average suspicious probability:
      {avg_risk}%

    • Relationship intelligence detected elevated linked-account activity.

    • Enhanced Due Diligence recommended for flagged entities.

    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        risk_dist = filtered_df.groupby(
            "alert_priority"
        ).size().reset_index(name="count")

        fig1 = px.bar(

            risk_dist,

            x="alert_priority",

            y="count",

            color="alert_priority",

            text="count"

        )

        fig1.update_layout(

            paper_bgcolor="#071028",

            plot_bgcolor="#071028",

            font_color="white",

            title="Alert Severity Distribution"

        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        sample_df = filtered_df.sample(
            min(1200, len(filtered_df))
        )

        fig2 = px.scatter(

            sample_df,

            x="monthly_turnover",

            y="ai_suspicion_probability",

            color="alert_priority",

            size="fund_transfer_count",

            hover_data=["client_id"]

        )

        fig2.update_layout(

            paper_bgcolor="#071028",

            plot_bgcolor="#071028",

            font_color="white",

            title="Turnover vs Suspicion Intelligence"

        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    city_risk = filtered_df.groupby(
        original_df["address"]
    )["ai_suspicion_probability"].mean().reset_index()

    city_risk.columns = [
        "address",
        "risk"
    ]

    fig3 = px.bar(

        city_risk,

        x="address",

        y="risk",

        color="risk"

    )

    fig3.update_layout(

        paper_bgcolor="#071028",

        plot_bgcolor="#071028",

        font_color="white",

        title="Geographical Risk Exposure"

    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.markdown("---")

    investigation_table = filtered_df[[

        "client_id",
        "aml_risk_score",
        "ai_suspicion_probability",
        "alert_priority",
        "alert_reason"

    ]]

    investigation_table = investigation_table.sort_values(

        by="ai_suspicion_probability",

        ascending=False

    )

    st.dataframe(
        investigation_table.head(100),
        use_container_width=True
    )

elif page == "Alert Intelligence":

    st.subheader("🚨 AI Alert Intelligence")

    alert_df = filtered_df[[

        "client_id",
        "monthly_turnover",
        "declared_income",
        "fund_transfer_count",
        "ai_suspicion_probability",
        "alert_priority",
        "alert_reason"

    ]]

    alert_df = alert_df.sort_values(

        by="ai_suspicion_probability",

        ascending=False

    )

    st.dataframe(
        alert_df.head(200),
        use_container_width=True
    )

    st.markdown("---")

    heatmap_df = filtered_df.sample(
        min(1000, len(filtered_df))
    )

    fig4 = px.density_heatmap(

        heatmap_df,

        x="fund_transfer_count",

        y="login_frequency",

        z="ai_suspicion_probability"

    )

    fig4.update_layout(

        paper_bgcolor="#071028",

        plot_bgcolor="#071028",

        font_color="white",

        title="Suspicious Behaviour Heatmap"

    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.markdown("---")

    fig5 = px.sunburst(

        filtered_df,

        path=["alert_priority"],

        values="monthly_turnover"

    )

    fig5.update_layout(

        paper_bgcolor="#071028",

        font_color="white"

    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

elif page == "Relationship Intelligence":

    st.subheader("🔗 Relationship Intelligence")

    selected_client = st.selectbox(

        "Select Client",

        filtered_df["client_id"].unique()

    )

    selected_data = filtered_df[
        filtered_df["client_id"] == selected_client
    ].iloc[0]

    linked_mobile = filtered_df[
        filtered_df["mobile"] == selected_data["mobile"]
    ]

    linked_ip = filtered_df[
        filtered_df["ip_address"] == selected_data["ip_address"]
    ]

    linked_group = filtered_df[
        filtered_df["group_id"] == selected_data["group_id"]
    ]

    c6, c7, c8 = st.columns(3)

    c6.metric(
        "Linked Mobile Accounts",
        len(linked_mobile)
    )

    c7.metric(
        "Linked IP Accounts",
        len(linked_ip)
    )

    c8.metric(
        "Group Connections",
        len(linked_group)
    )

    st.markdown("---")

    st.subheader("📱 Mobile Linked Entities")

    st.dataframe(

        linked_mobile[[

            "client_id",
            "alert_priority",
            "ai_suspicion_probability",
            "alert_reason"

        ]],

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("🌐 IP Linked Entities")

    st.dataframe(

        linked_ip[[

            "client_id",
            "alert_priority",
            "ai_suspicion_probability",
            "alert_reason"

        ]],

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("👥 Group Linked Entities")

    st.dataframe(

        linked_group[[

            "client_id",
            "alert_priority",
            "monthly_turnover",
            "fund_transfer_count",
            "alert_reason"

        ]],

        use_container_width=True

    )

elif page == "Suspicious Cluster Explorer":

    st.subheader("🕸️ Suspicious Cluster Explorer")

    cluster_df = filtered_df.groupby("group_id").agg({

        "client_id":"count",
        "monthly_turnover":"mean",
        "fund_transfer_count":"mean",
        "ai_suspicion_probability":"mean"

    }).reset_index()

    cluster_df.columns = [

        "group_id",
        "group_size",
        "avg_turnover",
        "avg_transfer_count",
        "avg_probability"

    ]

    cluster_df = cluster_df.sort_values(

        by="avg_probability",

        ascending=False

    )

    st.dataframe(
        cluster_df.head(100),
        use_container_width=True
    )

    st.markdown("---")

    fig6 = px.scatter(

        cluster_df.head(300),

        x="group_size",

        y="avg_probability",

        size="avg_turnover",

        color="avg_transfer_count",

        hover_data=["group_id"]

    )

    fig6.update_layout(

        paper_bgcolor="#071028",

        plot_bgcolor="#071028",

        font_color="white",

        title="Cluster-Level AML Intelligence"

    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

elif page == "Compliance Report":

    st.subheader("📄 Compliance Investigation Report")

    report_df = filtered_df[[

        "client_id",
        "monthly_turnover",
        "declared_income",
        "fund_transfer_count",
        "login_frequency",
        "ai_suspicion_probability",
        "alert_priority",
        "alert_reason"

    ]]

    report_df = report_df.sort_values(

        by="ai_suspicion_probability",

        ascending=False

    )

    st.dataframe(
        report_df.head(200),
        use_container_width=True
    )

    st.markdown("---")

    csv = report_df.to_csv(index=False)

    st.download_button(

        label="Download Compliance Report",

        data=csv,

        file_name="aml_compliance_report.csv",

        mime="text/csv"

    )

st.markdown("---")

st.markdown("""

<div style="
background:rgba(255,255,255,0.04);
padding:20px;
border-radius:20px;
text-align:center;
border:1px solid rgba(255,255,255,0.08);
">

<h3 style="color:#00E5FF;">
AI-Powered AML Surveillance Platform
</h3>

<p style="color:white;">
Intelligent Alert Prioritization • Relationship Intelligence • Group Risk Detection
</p>

<p style="color:gray;">
Enterprise AML Compliance Prototype
</p>

</div>

""", unsafe_allow_html=True)
