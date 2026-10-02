import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Industrial Machine AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FAILURE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "failure_association_results.csv"
)

TEMPORAL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "temporal_behavior_results.csv"
)

AI_PATH = os.path.join(
    BASE_DIR,
    "results",
    "ai_behavior_explanation.csv"
)

VALIDATION_PATH = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_validation_results.csv"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b1117;
    }

    [data-testid="stSidebar"] {
        background-color: #101820;
        border-right: 1px solid #26323d;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #f1f5f9;
    }

    [data-testid="stMetric"] {
        background-color: #111a23;
        border: 1px solid #26323d;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .section-title {
        font-size: 17px;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD FUNCTION
# =========================================================

@st.cache_data
def load_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


# =========================================================
# LOAD DATA
# =========================================================

failure_df = load_csv(FAILURE_PATH)
temporal_df = load_csv(TEMPORAL_PATH)
ai_df = load_csv(AI_PATH)
validation_df = load_csv(VALIDATION_PATH)


# =========================================================
# CHECK MAIN DATA
# =========================================================

if failure_df.empty:

    st.error(
        "failure_association_results.csv not found."
    )

    st.stop()


# =========================================================
# NORMALIZE COLUMN NAMES
# =========================================================

failure_df.columns = [
    str(c).strip().lower()
    for c in failure_df.columns
]

temporal_df.columns = [
    str(c).strip().lower()
    for c in temporal_df.columns
]

ai_df.columns = [
    str(c).strip().lower()
    for c in ai_df.columns
]

validation_df.columns = [
    str(c).strip().lower()
    for c in validation_df.columns
]


# =========================================================
# BASIC TYPES
# =========================================================

failure_df["engine_id"] = pd.to_numeric(
    failure_df["engine_id"],
    errors="coerce"
)

failure_df["cycle"] = pd.to_numeric(
    failure_df["cycle"],
    errors="coerce"
)


# =========================================================
# BEHAVIOUR CLUSTER
# =========================================================

if "behavior_cluster" in failure_df.columns:

    failure_df["cluster"] = pd.to_numeric(
        failure_df["behavior_cluster"],
        errors="coerce"
    )

elif "behavior_cluster" in temporal_df.columns:

    temporal_cluster = temporal_df[
        [
            "engine_id",
            "cycle",
            "behavior_cluster"
        ]
    ].drop_duplicates(
        subset=["engine_id", "cycle"]
    )

    failure_df = failure_df.merge(
        temporal_cluster,
        on=["engine_id", "cycle"],
        how="left"
    )

    failure_df["cluster"] = pd.to_numeric(
        failure_df["behavior_cluster"],
        errors="coerce"
    )

else:

    failure_df["cluster"] = 0


failure_df["cluster"] = (
    failure_df["cluster"]
    .fillna(0)
    .astype(int)
)


# =========================================================
# ANOMALY
# =========================================================

if "anomaly" not in failure_df.columns:

    if "anomaly_prediction" in failure_df.columns:

        failure_df["anomaly"] = (
            pd.to_numeric(
                failure_df["anomaly_prediction"],
                errors="coerce"
            )
            .fillna(0)
        )

    else:

        failure_df["anomaly"] = 0


failure_df["anomaly"] = (
    pd.to_numeric(
        failure_df["anomaly"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# =========================================================
# LIFE PROGRESS
# =========================================================

if "life_progress" not in failure_df.columns:

    max_cycle = (
        failure_df
        .groupby("engine_id")["cycle"]
        .transform("max")
    )

    failure_df["life_progress"] = (
        failure_df["cycle"] / max_cycle
    )


failure_df["life_progress"] = (
    pd.to_numeric(
        failure_df["life_progress"],
        errors="coerce"
    )
    .fillna(0)
    .clip(0, 1)
)


# =========================================================
# LIFE STAGE
# =========================================================

if "life_stage" not in failure_df.columns:

    failure_df["life_stage"] = np.where(
        failure_df["life_progress"] <= 0.33,
        "Early",
        np.where(
            failure_df["life_progress"] <= 0.66,
            "Middle",
            "Late"
        )
    )


# =========================================================
# SORT DATA
# =========================================================

failure_df = failure_df.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)


# =========================================================
# GLOBAL VALUES
# =========================================================

total_machines = int(
    failure_df["engine_id"].nunique()
)

total_observations = int(
    len(failure_df)
)

total_anomalies = int(
    failure_df["anomaly"].sum()
)

total_behaviours = int(
    failure_df["cluster"].nunique()
)


# =========================================================
# MACHINE LIST
# =========================================================

machine_ids = sorted(
    failure_df["engine_id"]
    .dropna()
    .astype(int)
    .unique()
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## ⚙️ INDUSTRIAL MACHINE AI"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "SYSTEM",
    [
        "Dashboard",
        "Live Monitoring",
        "Behaviour Discovery",
        "Anomaly Analysis",
        "Historical Analysis",
        "AI Insights",
        "Alerts"
    ]
)

st.sidebar.markdown("---")

selected_machine = st.sidebar.selectbox(
    "MACHINE",
    machine_ids,
    format_func=lambda x:
        f"MACHINE {x:02d}"
)


# =========================================================
# SELECTED MACHINE
# =========================================================

machine_df = failure_df[
    failure_df["engine_id"] == selected_machine
].copy()

machine_df = machine_df.sort_values(
    "cycle"
)


if machine_df.empty:
    st.stop()


latest = machine_df.iloc[-1]


# =========================================================
# HEADER
# =========================================================

st.title("Industrial Machine AI")

st.markdown("---")


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    # -----------------------------------------------------
    # KPI
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MACHINES MONITORED",
        total_machines
    )

    c2.metric(
        "OBSERVATIONS",
        f"{total_observations:,}"
    )

    c3.metric(
        "ANOMALIES",
        total_anomalies
    )

    c4.metric(
        "BEHAVIOUR STATES",
        total_behaviours
    )

    st.markdown("---")

    # -----------------------------------------------------
    # MACHINE
    # -----------------------------------------------------

    c1, c2 = st.columns([3, 1])

    with c1:

        st.subheader(
            f"MACHINE {selected_machine:02d}"
        )

        st.caption(
            "INDUSTRIAL UNIT"
        )

    with c2:

        if int(latest["anomaly"]) == 1:

            st.metric(
                "STATUS",
                "ANOMALY"
            )

        else:

            st.metric(
                "STATUS",
                "MONITORING"
            )

    # -----------------------------------------------------
    # MACHINE KPI
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    current_cycle = int(
        latest["cycle"]
    )

    life_progress = (
        float(latest["life_progress"])
        * 100
    )

    behaviour_state = int(
        latest["cluster"]
    )

    anomaly_count = int(
        machine_df["anomaly"].sum()
    )

    c1.metric(
        "CURRENT CYCLE",
        current_cycle
    )

    c2.metric(
        "LIFE PROGRESS",
        f"{life_progress:.1f}%"
    )

    c3.metric(
        "BEHAVIOUR",
        f"STATE {behaviour_state}"
    )

    c4.metric(
        "ANOMALY COUNT",
        anomaly_count
    )

    # -----------------------------------------------------
    # SENSOR MONITORING
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'SENSOR MONITORING'
        '</div>',
        unsafe_allow_html=True
    )

    sensor_columns = [
        c for c in machine_df.columns
        if c.startswith("sensor_")
        and not c.endswith("_diff")
        and not c.endswith("_rolling_mean")
        and not c.endswith("_rolling_std")
    ]

    if sensor_columns:

        selected_sensor = st.selectbox(
            "SENSOR",
            sensor_columns,
            label_visibility="collapsed"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=machine_df["cycle"],
                y=machine_df[selected_sensor],
                mode="lines",
                name=selected_sensor
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            xaxis_title="Cycle",
            yaxis_title=selected_sensor
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # BEHAVIOUR TIMELINE
    # -----------------------------------------------------

    st.markdown("### BEHAVIOUR TIMELINE")

    timeline_df = machine_df[
        ["cycle", "cluster"]
    ].copy()

    timeline_df["cycle"] = pd.to_numeric(
        timeline_df["cycle"],
        errors="coerce"
    )

    timeline_df["cluster"] = pd.to_numeric(
        timeline_df["cluster"],
        errors="coerce"
    )

    timeline_df = timeline_df.dropna()

    if not timeline_df.empty:

        timeline_fig = px.line(
            timeline_df,
            x="cycle",
            y="cluster",
            markers=True
        )

        timeline_fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            xaxis_title="Cycle",
            yaxis_title="Behaviour State",
            showlegend=False
        )

        timeline_fig.update_yaxes(
            dtick=1
        )

        st.plotly_chart(
            timeline_fig,
            use_container_width=True,
            key="behaviour_timeline"
        )

    else:

        st.info(
            "No timeline data available."
        )


# =========================================================
# LIVE MONITORING
# =========================================================

elif page == "Live Monitoring":

    st.subheader(
        f"MACHINE {selected_machine:02d}"
    )

    st.caption(
        "CURRENT MACHINE STATE"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "CURRENT CYCLE",
        int(latest["cycle"])
    )

    c2.metric(
        "LIFE PROGRESS",
        f"{float(latest['life_progress']) * 100:.1f}%"
    )

    c3.metric(
        "BEHAVIOUR",
        f"STATE {int(latest['cluster'])}"
    )

    if int(latest["anomaly"]) == 1:

        c4.metric(
            "STATUS",
            "ANOMALY"
        )

    else:

        c4.metric(
            "STATUS",
            "MONITORING"
        )

    st.markdown("---")

    sensor_columns = [
        c for c in machine_df.columns
        if c.startswith("sensor_")
        and not c.endswith("_diff")
        and not c.endswith("_rolling_mean")
        and not c.endswith("_rolling_std")
    ]

    if sensor_columns:

        sensor = st.selectbox(
            "SENSOR",
            sensor_columns
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=machine_df["cycle"],
                y=machine_df[sensor],
                mode="lines",
                name=sensor
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Cycle",
            yaxis_title=sensor
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# BEHAVIOUR DISCOVERY
# =========================================================

elif page == "Behaviour Discovery":

    st.subheader(
        "BEHAVIOUR DISCOVERY"
    )

    c1, c2 = st.columns(2)

    # -----------------------------------------------------
    # BEHAVIOUR DISTRIBUTION
    # -----------------------------------------------------

    behaviour_counts = (
        failure_df["cluster"]
        .value_counts()
        .sort_index()
    )

    with c1:

        fig = px.bar(
            x=[
                f"STATE {int(x)}"
                for x in behaviour_counts.index
            ],
            y=behaviour_counts.values,
            height=400
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Behaviour State",
            yaxis_title="Observations"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # ANOMALIES BY BEHAVIOUR
    # -----------------------------------------------------

    with c2:

        cluster_anomaly = (
            failure_df
            .groupby("cluster")["anomaly"]
            .sum()
            .reset_index()
        )

        cluster_anomaly["state"] = (
            cluster_anomaly["cluster"]
            .apply(
                lambda x:
                f"STATE {int(x)}"
            )
        )

        fig = px.bar(
            cluster_anomaly,
            x="state",
            y="anomaly",
            height=400
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Behaviour State",
            yaxis_title="Anomalies"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # LIFE PROGRESS VS BEHAVIOUR
    # -----------------------------------------------------

    behaviour_map = failure_df[
        [
            "engine_id",
            "cycle",
            "life_progress",
            "cluster",
            "anomaly"
        ]
    ].copy()

    fig = px.scatter(
        behaviour_map,
        x="life_progress",
        y="cluster",
        color="cluster",
        hover_data=[
            "engine_id",
            "cycle",
            "anomaly"
        ],
        height=450
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Life Progress",
        yaxis_title="Behaviour State"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ANOMALY ANALYSIS
# =========================================================

elif page == "Anomaly Analysis":

    st.subheader(
        "ANOMALY ANALYSIS"
    )

    anomaly_df = failure_df[
        failure_df["anomaly"] == 1
    ].copy()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "TOTAL ANOMALIES",
        total_anomalies
    )

    anomaly_rate = (
        total_anomalies /
        total_observations *
        100
    )

    c2.metric(
        "ANOMALY RATE",
        f"{anomaly_rate:.2f}%"
    )

    affected_machines = (
        anomaly_df["engine_id"]
        .nunique()
    )

    c3.metric(
        "AFFECTED MACHINES",
        affected_machines
    )

    if not anomaly_df.empty:

        max_count = (
            anomaly_df
            .groupby("engine_id")
            .size()
            .max()
        )

    else:

        max_count = 0

    c4.metric(
        "MAX MACHINE ANOMALIES",
        int(max_count)
    )

    st.markdown("---")

    # -----------------------------------------------------
    # ANOMALY TIMELINE
    # -----------------------------------------------------

    if not anomaly_df.empty:

        fig = px.scatter(
            anomaly_df,
            x="cycle",
            y="engine_id",
            color="cluster",
            hover_data=[
                "cycle",
                "cluster",
                "life_progress"
            ],
            height=500
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Cycle",
            yaxis_title="Machine"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # MACHINE ANOMALY COUNT
    # -----------------------------------------------------

    machine_anomaly_counts = (
        anomaly_df
        .groupby("engine_id")
        .size()
        .reset_index(
            name="anomaly_count"
        )
        .sort_values(
            "anomaly_count",
            ascending=False
        )
    )

    st.dataframe(
        machine_anomaly_counts,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# HISTORICAL ANALYSIS
# =========================================================

elif page == "Historical Analysis":

    st.subheader(
        "HISTORICAL ANALYSIS"
    )

    stage_counts = (
        failure_df["life_stage"]
        .value_counts()
        .reindex(
            [
                "Early",
                "Middle",
                "Late"
            ]
        )
        .fillna(0)
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "EARLY",
        int(stage_counts["Early"])
    )

    c2.metric(
        "MIDDLE",
        int(stage_counts["Middle"])
    )

    c3.metric(
        "LATE",
        int(stage_counts["Late"])
    )

    st.markdown("---")

    # -----------------------------------------------------
    # ANOMALY RATE BY LIFE STAGE
    # -----------------------------------------------------

    stage_anomaly = (
        failure_df
        .groupby("life_stage")["anomaly"]
        .agg(
            ["count", "sum"]
        )
        .reset_index()
    )

    stage_anomaly["anomaly_rate"] = (
        stage_anomaly["sum"] /
        stage_anomaly["count"] *
        100
    )

    stage_anomaly = stage_anomaly[
        stage_anomaly["life_stage"].isin(
            [
                "Early",
                "Middle",
                "Late"
            ]
        )
    ]

    fig = px.bar(
        stage_anomaly,
        x="life_stage",
        y="anomaly_rate",
        height=400
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Life Stage",
        yaxis_title="Anomaly Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # BEHAVIOUR BY LIFE STAGE
    # -----------------------------------------------------

    behaviour_stage = pd.crosstab(
        failure_df["life_stage"],
        failure_df["cluster"],
        normalize="index"
    ) * 100

    behaviour_stage = behaviour_stage.reindex(
        [
            "Early",
            "Middle",
            "Late"
        ]
    )

    fig = px.bar(
        behaviour_stage,
        barmode="stack",
        height=450
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Life Stage",
        yaxis_title="Percentage (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# AI INSIGHTS
# =========================================================

elif page == "AI Insights":

    st.subheader(
        "AI INSIGHTS"
    )

    # -----------------------------------------------------
    # IMPORTANT FEATURES
    # -----------------------------------------------------

    if not ai_df.empty:

        feature_candidates = [
            c for c in ai_df.columns
            if (
                "feature" in c.lower()
                or "sensor" in c.lower()
            )
        ]

        numeric_candidates = [
            c for c in ai_df.columns
            if pd.api.types.is_numeric_dtype(
                ai_df[c]
            )
        ]

        if (
            feature_candidates
            and numeric_candidates
        ):

            feature_column = (
                feature_candidates[0]
            )

            value_candidates = [
                c for c in numeric_candidates
                if c != feature_column
            ]

            if value_candidates:

                value_column = (
                    value_candidates[-1]
                )

                insight_df = ai_df[
                    [
                        feature_column,
                        value_column
                    ]
                ].dropna()

                insight_df = (
                    insight_df
                    .sort_values(
                        value_column,
                        ascending=False
                    )
                    .head(10)
                )

                fig = px.bar(
                    insight_df,
                    x=value_column,
                    y=feature_column,
                    orientation="h",
                    height=500
                )

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Difference",
                    yaxis_title="Feature"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.dataframe(
                    ai_df,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.dataframe(
                ai_df,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.dataframe(
            failure_df.head(50),
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # KMEANS VALIDATION
    # -----------------------------------------------------

    st.markdown("---")

    if not validation_df.empty:

        k_column = None
        silhouette_column = None

        for column in validation_df.columns:

            name = column.lower()

            if name in [
                "k",
                "clusters",
                "n_clusters"
            ]:

                k_column = column

            if "silhouette" in name:

                silhouette_column = column

        if (
            k_column is not None
            and
            silhouette_column is not None
        ):

            fig = px.line(
                validation_df,
                x=k_column,
                y=silhouette_column,
                markers=True,
                height=400
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Number of Clusters",
                yaxis_title="Silhouette Score"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# ALERTS
# =========================================================

elif page == "Alerts":

    st.subheader(
        "ALERTS"
    )

    alert_df = failure_df[
        failure_df["anomaly"] == 1
    ].copy()

    if alert_df.empty:

        st.dataframe(
            pd.DataFrame(
                columns=[
                    "MACHINE",
                    "CYCLE",
                    "BEHAVIOUR",
                    "LIFE",
                    "STATUS"
                ]
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        alert_output = pd.DataFrame()

        alert_output["MACHINE"] = (
            alert_df["engine_id"]
            .astype(int)
            .apply(
                lambda x:
                f"MACHINE {x:02d}"
            )
        )

        alert_output["CYCLE"] = (
            alert_df["cycle"]
            .astype(int)
        )

        alert_output["BEHAVIOUR"] = (
            alert_df["cluster"]
            .astype(int)
            .apply(
                lambda x:
                f"STATE {x}"
            )
        )

        alert_output["LIFE"] = (
            alert_df["life_progress"] * 100
        ).round(1).astype(str) + "%"

        alert_output["STATUS"] = (
            "ANOMALY"
        )

        alert_output = (
            alert_output
            .sort_values(
                "CYCLE",
                ascending=False
            )
        )

        st.dataframe(
            alert_output,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Industrial Machine AI"
)