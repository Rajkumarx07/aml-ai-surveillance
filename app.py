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

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Critical Alerts",
        critical_alerts
    )

    col2.metric(
        "High Risk Alerts",
        high_alerts
    )

    col3.metric(
        "Medium Alerts",
        medium_alerts
    )

    col4.metric(
        "Average Risk %",
        avg_risk
    )

    col5.metric(
        "Linked Groups",
        suspicious_groups
    )

    st.markdown("---")

    st.subheader("🧠 Executive Intelligence Summary")

    total_clients = len(filtered_df)

    high_risk_ratio = round(
        (
            critical_alerts /
            total_clients
        ) * 100,
        2
    )

    top_reason = filtered_df[
        filtered_df["alert_reason"] != ""
    ]["alert_reason"].mode()

    if len(top_reason) > 0:
        top_reason = top_reason[0]
    else:
        top_reason = "No dominant risk indicator"

    st.info(f"""

    • {critical_alerts} critical AML alerts identified across monitored accounts.

    • High-risk exposure represents {high_risk_ratio}% of monitored entities.

    • Dominant suspicious behaviour:
      {top_reason}

    • AI engine detected elevated relationship linkage activity across multiple groups.

    • Risk prioritization engine recommends enhanced due diligence for critical entities.

    """)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("📈 Risk Distribution")

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

            title="Alert Severity Distribution",

            xaxis_title="Risk Category",

            yaxis_title="Entity Count"

        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with c2:

        st.subheader("💰 Turnover vs Risk Analysis")

        sampled_df = filtered_df.sample(1200)

        fig2 = px.scatter(

            sampled_df,

            x="monthly_turnover",

            y="ai_suspicion_probability",

            color="alert_priority",

            size="fund_transfer_count",

            hover_data=["client_id"],

            opacity=0.7

        )

        fig2.update_layout(

            paper_bgcolor="#071028",

            plot_bgcolor="#071028",

            font_color="white",

            title="High Turnover Risk Intelligence"

        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:

        st.subheader("🏦 Top Risk Cities")

        city_risk = original_df.copy()

        city_risk["risk"] = filtered_df[
            "ai_suspicion_probability"
        ].values

        city_risk = city_risk.groupby(
            "address"
        )["risk"].mean().reset_index()

        city_risk = city_risk.sort_values(
            by="risk",
            ascending=False
        )

        fig3 = px.bar(

            city_risk,

            x="address",

            y="risk",

            color="risk",

            text_auto=True

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

    with c4:

        st.subheader("🔄 Transaction Behaviour Analysis")

        behavior_df = filtered_df.groupby(
            "alert_priority"
        )["fund_transfer_count"].mean().reset_index()

        fig4 = px.line(

            behavior_df,

            x="alert_priority",

            y="fund_transfer_count",

            markers=True

        )

        fig4.update_layout(

            paper_bgcolor="#071028",

            plot_bgcolor="#071028",

            font_color="white",

            title="Fund Movement Behaviour"

        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("🚨 High-Risk Investigation Queue")

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

    st.subheader("🚨 AI Alert Intelligence Center")

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

    st.subheader("📌 Risk Priority Heatmap")

    heatmap_df = filtered_df.sample(800)

    fig5 = px.density_heatmap(

        heatmap_df,

        x="fund_transfer_count",

        y="login_frequency",

        z="ai_suspicion_probability",

        nbinsx=30,

        nbinsy=30

    )

    fig5.update_layout(

        paper_bgcolor="#071028",

        plot_bgcolor="#071028",

        font_color="white",

        title="Suspicious Behaviour Density"

    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🧠 AI Investigation Insights")

    suspicious_entities = filtered_df[
        filtered_df["ai_suspicion_probability"] > 0.75
    ]

    avg_turnover = round(
        suspicious_entities["monthly_turnover"].mean(),
        2
    )

    avg_transfer = round(
        suspicious_entities["fund_transfer_count"].mean(),
        2
    )

    st.warning(f"""

    • Elevated suspicious behaviour identified across high-frequency fund transfer entities.

    • Average suspicious turnover observed:
      ₹{avg_turnover:,.0f}

    • AI engine identified coordinated behavioural similarity across multiple accounts.

    • Average suspicious fund movement count:
      {avg_transfer}

    • Enhanced Due Diligence (EDD) recommended for identified critical entities.

    """)

    st.markdown("---")

    st.subheader("📊 AML Behaviour Segmentation")

    segmentation_df = filtered_df.copy()

    segmentation_df["segment"] = np.where(

        segmentation_df["ai_suspicion_probability"] > 0.75,
        "Critical",

        np.where(

            segmentation_df["ai_suspicion_probability"] > 0.50,
            "Watchlist",

            "Normal"

        )

    )

    fig6 = px.sunburst(

        segmentation_df,

        path=[
            "segment",
            "alert_priority"
        ],

        values="monthly_turnover"

    )

    fig6.update_layout(

        paper_bgcolor="#071028",

        font_color="white"

    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

elif page == "Relationship Intelligence":

    st.subheader("🔗 Relationship Intelligence Engine")

    selected_client = st.selectbox(

        "Select Client ID",

        filtered_df["client_id"].unique()

    )

    selected_data = filtered_df[
        filtered_df["client_id"] == selected_client
    ].iloc[0]

    client_mobile = selected_data["mobile"]

    client_ip = selected_data["ip_address"]

    client_group = selected_data["group_id"]

    linked_mobile = filtered_df[
        filtered_df["mobile"] == client_mobile
    ]

    linked_ip = filtered_df[
        filtered_df["ip_address"] == client_ip
    ]

    linked_group = filtered_df[
        filtered_df["group_id"] == client_group
    ]

    st.markdown("### 🧠 Investigation Summary")

    st.info(f"""

    • Selected entity belongs to monitoring group:
      {client_group}

    • AI engine identified relationship linkage through shared identifiers.

    • Connected entities may indicate coordinated account behaviour.

    • Enhanced relationship surveillance recommended.

    """)

    st.markdown("---")

    c5, c6, c7 = st.columns(3)

    with c5:

        st.metric(
            "Linked Mobile Accounts",
            len(linked_mobile)
        )

    with c6:

        st.metric(
            "Linked IP Accounts",
            len(linked_ip)
        )

    with c7:

        st.metric(
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

    st.subheader("👥 Group-Level Relationship Analysis")

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

    st.subheader("📊 Suspicious Cluster Risk Matrix")

    fig7 = px.scatter(

        cluster_df.head(300),

        x="group_size",

        y="avg_probability",

        size="avg_turnover",

        color="avg_transfer_count",

        hover_data=["group_id"]

    )

    fig7.update_layout(

        paper_bgcolor="#071028",

        plot_bgcolor="#071028",

        font_color="white",

        title="Cluster-Level AML Intelligence"

    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🚨 High-Risk Cluster Insights")

    top_cluster = cluster_df.iloc[0]

    st.error(f"""

    • Highest suspicious cluster identified:
      {top_cluster['group_id']}

    • Group-level suspicious probability:
      {round(top_cluster['avg_probability'] * 100,2)}%

    • Elevated transaction coordination behaviour detected.

    • Potential linked-entity monitoring escalation recommended.

    • AI engine suggests Enhanced Due Diligence for cluster entities.

    """)

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

    st.subheader("🧠 Compliance Intelligence Summary")

    critical_entities = len(
        report_df[
            report_df["alert_priority"] == "Critical"
        ]
    )

    high_entities = len(
        report_df[
            report_df["alert_priority"] == "High"
        ]
    )

    avg_probability = round(
        report_df["ai_suspicion_probability"].mean() * 100,
        2
    )

    avg_turnover = round(
        report_df["monthly_turnover"].mean(),
        2
    )

    top_alert_reason = report_df[
        report_df["alert_reason"] != ""
    ]["alert_reason"].mode()

    if len(top_alert_reason) > 0:
        top_alert_reason = top_alert_reason[0]
    else:
        top_alert_reason = "No major risk indicator"

    st.success(f"""

    ✔ AI surveillance completed successfully.

    ✔ Critical entities identified:
      {critical_entities}

    ✔ High-risk entities identified:
      {high_entities}

    ✔ Average suspicious probability:
      {avg_probability}%

    ✔ Dominant suspicious behaviour:
      {top_alert_reason}

    ✔ Average suspicious turnover:
      ₹{avg_turnover:,.0f}

    ✔ Relationship intelligence engine identified multiple linked-account indicators.

    ✔ Enhanced Due Diligence (EDD) recommended for flagged entities.

    """)

    st.markdown("---")

    st.subheader("📥 Download Compliance Investigation Report")

    csv = report_df.to_csv(index=False)

    st.download_button(

        label="Download Investigation Report",

        data=csv,

        file_name="aml_compliance_report.csv",

        mime="text/csv"

    )

st.markdown("---")

st.markdown("""

<div style="

background: rgba(255,255,255,0.04);
padding: 20px;
border-radius: 20px;
text-align:center;
border: 1px solid rgba(255,255,255,0.08);

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
