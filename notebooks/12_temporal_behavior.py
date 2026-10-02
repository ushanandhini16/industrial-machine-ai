import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# -----------------------------
# 1. Load Dataset
# -----------------------------

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

# -----------------------------
# 2. Remove Constant Columns
# -----------------------------

constant_columns = df.columns[df.nunique() == 1]

df = df.drop(columns=constant_columns)

# -----------------------------
# 3. Feature Engineering
# -----------------------------

sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

for sensor in sensor_columns:

    # Difference from previous cycle
    df[sensor + "_diff"] = (
        df.groupby("engine_id")[sensor].diff().fillna(0)
    )

    # Rolling mean
    df[sensor + "_rolling_mean"] = (
        df.groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # Rolling standard deviation
    df[sensor + "_rolling_std"] = (
        df.groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    )

# -----------------------------
# 4. Select ML Features
# -----------------------------

feature_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_columns]

print("Feature Shape:", X.shape)

# -----------------------------
# 5. Standardization
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------
# 6. Behaviour Clustering
# -----------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["behavior_cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# 7. Anomaly Detection
# -----------------------------

isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

df["anomaly_prediction"] = isolation_forest.fit_predict(X_scaled)

df["anomaly"] = df["anomaly_prediction"].apply(
    lambda x: 1 if x == -1 else 0
)

# -----------------------------
# 8. Sort by Engine and Cycle
# -----------------------------

df = df.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

# -----------------------------
# 9. Previous Behaviour
# -----------------------------

df["previous_cluster"] = (
    df.groupby("engine_id")["behavior_cluster"].shift(1)
)

# -----------------------------
# 10. Detect Behaviour Changes
# -----------------------------

df["cluster_changed"] = (
    df["behavior_cluster"] != df["previous_cluster"]
).astype(int)

# First cycle of every engine is not a transition
df.loc[df["previous_cluster"].isna(), "cluster_changed"] = 0

# -----------------------------
# 11. Behaviour Transitions
# -----------------------------

transitions = df[df["cluster_changed"] == 1].copy()

print("\nTotal Behaviour Transitions:")
print(len(transitions))

print("\nBehaviour Transition Counts:")

transition_counts = (
    transitions
    .groupby(["previous_cluster", "behavior_cluster"])
    .size()
    .sort_values(ascending=False)
)

print(transition_counts)

# -----------------------------
# 12. Anomalies Near Behaviour Changes
# -----------------------------

print("\nAnomalies During Behaviour Changes:")

transition_anomalies = transitions["anomaly"].sum()

print("Behaviour Changes:", len(transitions))
print("Anomalies During Changes:", transition_anomalies)

if len(transitions) > 0:
    percentage = (
        transition_anomalies / len(transitions)
    ) * 100

    print(
        "Anomaly Percentage During Behaviour Changes:",
        round(percentage, 2),
        "%"
    )

# -----------------------------
# 13. Sample Temporal Patterns
# -----------------------------

print("\nSample Behaviour Sequences:")

for engine in df["engine_id"].unique()[:5]:

    engine_data = df[df["engine_id"] == engine]

    sequence = engine_data[
        ["cycle", "behavior_cluster", "anomaly"]
    ].head(20)

    print("\nEngine:", engine)
    print(sequence.to_string(index=False))

# -----------------------------
# 14. Save Results
# -----------------------------

output_path = (
    r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI"
    r"\results\temporal_behavior_results.csv"
)

df.to_csv(output_path, index=False)

print("\nResults saved successfully!")
print(output_path)