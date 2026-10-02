import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# --------------------------------
# 1. Load Dataset
# --------------------------------

file_path = r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI\data\CMAPSSData\train_FD001.txt"

columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4",
    "sensor_5", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_10", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_16",
    "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

# --------------------------------
# 2. Remove Constant Columns
# --------------------------------

constant_columns = df.columns[df.nunique() == 1]

df = df.drop(columns=constant_columns)

# --------------------------------
# 3. Sensor Columns
# --------------------------------

sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

# --------------------------------
# 4. Feature Engineering
# --------------------------------

for sensor in sensor_columns:

    df[sensor + "_diff"] = (
        df.groupby("engine_id")[sensor]
        .diff()
        .fillna(0)
    )

    df[sensor + "_rolling_mean"] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                5,
                min_periods=1
            ).mean()
        )
    )

    df[sensor + "_rolling_std"] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                5,
                min_periods=1
            ).std().fillna(0)
        )
    )

# --------------------------------
# 5. ML Features
# --------------------------------

feature_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_columns]

print("Feature Shape:", X.shape)

# --------------------------------
# 6. Scaling
# --------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------
# 7. Behaviour Clustering
# --------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["behavior_cluster"] = kmeans.fit_predict(X_scaled)

# --------------------------------
# 8. Anomaly Detection
# --------------------------------

isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

df["anomaly_prediction"] = (
    isolation_forest.fit_predict(X_scaled)
)

df["anomaly"] = (
    df["anomaly_prediction"]
    .apply(lambda x: 1 if x == -1 else 0)
)

# --------------------------------
# 9. Maximum Life of Each Engine
# --------------------------------

max_cycle = (
    df.groupby("engine_id")["cycle"]
    .max()
    .rename("max_cycle")
)

df = df.merge(
    max_cycle,
    on="engine_id"
)

# --------------------------------
# 10. Life Progress
# --------------------------------

df["life_progress"] = (
    df["cycle"] / df["max_cycle"]
)

# --------------------------------
# 11. Life Stage
# --------------------------------

def get_life_stage(progress):

    if progress <= 0.33:
        return "Early Life"

    elif progress <= 0.66:
        return "Middle Life"

    else:
        return "Late Life"


df["life_stage"] = (
    df["life_progress"]
    .apply(get_life_stage)
)

# --------------------------------
# 12. Overall Anomaly Rate
# --------------------------------

print("\nOverall Anomaly Rate:")

overall_anomaly_rate = (
    df["anomaly"].mean() * 100
)

print(
    round(overall_anomaly_rate, 2),
    "%"
)

# --------------------------------
# 13. Anomaly Rate by Life Stage
# --------------------------------

print("\nAnomaly Rate by Life Stage:")

stage_analysis = (
    df.groupby("life_stage")
    .agg(
        observations=("anomaly", "count"),
        anomalies=("anomaly", "sum"),
        anomaly_rate=("anomaly", "mean")
    )
)

stage_analysis["anomaly_rate"] = (
    stage_analysis["anomaly_rate"] * 100
)

print(stage_analysis)

# --------------------------------
# 14. Behaviour Cluster by Life Stage
# --------------------------------

print("\nBehaviour Cluster Distribution by Life Stage:")

cluster_stage = pd.crosstab(
    df["life_stage"],
    df["behavior_cluster"],
    normalize="index"
) * 100

print(
    cluster_stage.round(2)
)

# --------------------------------
# 15. Anomaly Rate by Cluster
# --------------------------------

print("\nAnomaly Rate by Behaviour Cluster:")

cluster_analysis = (
    df.groupby("behavior_cluster")
    .agg(
        observations=("anomaly", "count"),
        anomalies=("anomaly", "sum"),
        anomaly_rate=("anomaly", "mean"),
        average_life_progress=("life_progress", "mean")
    )
)

cluster_analysis["anomaly_rate"] = (
    cluster_analysis["anomaly_rate"] * 100
)

print(
    cluster_analysis.round(2)
)

# --------------------------------
# 16. Late-Life Behaviour Analysis
# --------------------------------

late_life = df[
    df["life_stage"] == "Late Life"
]

print("\nLate Life Analysis:")

print(
    "Late Life Observations:",
    len(late_life)
)

print(
    "Late Life Anomalies:",
    late_life["anomaly"].sum()
)

if len(late_life) > 0:

    late_rate = (
        late_life["anomaly"].mean() * 100
    )

    print(
        "Late Life Anomaly Rate:",
        round(late_rate, 2),
        "%"
    )

# --------------------------------
# 17. Most Unusual Late-Life Observations
# --------------------------------

print("\nLate-Life Anomalies:")

late_anomalies = late_life[
    late_life["anomaly"] == 1
][
    [
        "engine_id",
        "cycle",
        "max_cycle",
        "life_progress",
        "behavior_cluster"
    ]
]

print(
    late_anomalies
    .sort_values(
        "life_progress",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

# --------------------------------
# 18. Save Results
# --------------------------------

output_path = (
    r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI"
    r"\results\failure_association_results.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nResults saved successfully!")

print(output_path)