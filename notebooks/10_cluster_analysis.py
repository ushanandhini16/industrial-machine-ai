import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Load dataset
file_path = r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI\data\CMAPSSData\train_FD001.txt"

columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

# Remove constant columns
constant_columns = [
    col for col in df.columns
    if df[col].nunique() <= 1
]

df = df.drop(columns=constant_columns)

# Sensor columns
sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

# Sort
df = df.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

# Feature engineering
for sensor in sensor_columns:

    df[sensor + "_diff"] = (
        df.groupby("engine_id")[sensor].diff()
    )

    df[sensor + "_rolling_mean"] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                window=5,
                min_periods=1
            ).mean()
        )
    )

    df[sensor + "_rolling_std"] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                window=5,
                min_periods=1
            ).std()
        )
    )

df = df.fillna(0)

# ML features
feature_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_columns]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["behavior_cluster"] = kmeans.fit_predict(X_scaled)

# Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(X)

df["anomaly"] = model.predict(X)

df["anomaly_label"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# -----------------------------------
# 1. Cluster size
# -----------------------------------

print("\nCluster Size:")
print(
    df["behavior_cluster"]
    .value_counts()
    .sort_index()
)

# -----------------------------------
# 2. Anomalies in each cluster
# -----------------------------------

print("\nAnomalies by Cluster:")

cluster_anomalies = pd.crosstab(
    df["behavior_cluster"],
    df["anomaly_label"]
)

print(cluster_anomalies)

# -----------------------------------
# 3. Anomaly percentage
# -----------------------------------

anomaly_percentage = (
    df.groupby("behavior_cluster")["anomaly"]
    .apply(lambda x: (x == -1).mean() * 100)
)

print("\nAnomaly Percentage by Cluster:")
print(anomaly_percentage.round(2))

# -----------------------------------
# 4. Average sensor values
# -----------------------------------

cluster_sensor_mean = (
    df.groupby("behavior_cluster")[sensor_columns]
    .mean()
)

print("\nAverage Sensor Values by Cluster:")
print(cluster_sensor_mean.round(2))

# -----------------------------------
# 5. Average cycle
# -----------------------------------

print("\nAverage Cycle by Cluster:")

print(
    df.groupby("behavior_cluster")["cycle"]
    .mean()
    .round(2)
)

# -----------------------------------
# 6. Minimum and maximum cycle
# -----------------------------------

print("\nCycle Range by Cluster:")

cycle_range = df.groupby(
    "behavior_cluster"
)["cycle"].agg(["min", "max"])

print(cycle_range)