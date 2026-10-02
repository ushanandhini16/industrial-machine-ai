import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score

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
# 7. Isolation Forest
# --------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

df["anomaly_prediction"] = model.fit_predict(X_scaled)

df["anomaly"] = (
    df["anomaly_prediction"]
    .apply(lambda x: 1 if x == -1 else 0)
)

# --------------------------------
# 8. Calculate Remaining Life
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

df["remaining_cycles"] = (
    df["max_cycle"] - df["cycle"]
)

# --------------------------------
# 9. Define Near-Failure Window
# --------------------------------

# Last 20% of each engine's life

df["near_failure"] = (
    df["remaining_cycles"]
    <= (df["max_cycle"] * 0.20)
).astype(int)

# --------------------------------
# 10. Evaluation
# --------------------------------

y_true = df["near_failure"]
y_pred = df["anomaly"]

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

print("\nNear-Failure Detection Performance:")

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)

# --------------------------------
# 11. Confusion Matrix
# --------------------------------

confusion = pd.crosstab(
    y_true,
    y_pred,
    rownames=["Actual"],
    colnames=["Predicted"]
)

print("\nConfusion Matrix:")
print(confusion)

# --------------------------------
# 12. Anomaly Distribution
# --------------------------------

print("\nAnomaly Distribution:")

anomaly_distribution = (
    df.groupby("near_failure")["anomaly"]
    .agg(
        observations="count",
        anomalies="sum"
    )
)

anomaly_distribution["anomaly_rate"] = (
    anomaly_distribution["anomalies"]
    / anomaly_distribution["observations"]
    * 100
)

print(
    anomaly_distribution.round(2)
)

# --------------------------------
# 13. Distance From Failure
# --------------------------------

anomalies = df[
    df["anomaly"] == 1
].copy()

print("\nAnomaly Distance From Failure:")

print(
    anomalies["remaining_cycles"]
    .describe()
)

# --------------------------------
# 14. Closest Anomalies
# --------------------------------

print("\nClosest Anomalies To Failure:")

closest = anomalies[
    [
        "engine_id",
        "cycle",
        "max_cycle",
        "remaining_cycles"
    ]
].sort_values(
    "remaining_cycles"
).head(20)

print(
    closest.to_string(index=False)
)

# --------------------------------
# 15. Behaviour Clustering
# --------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["behavior_cluster"] = (
    kmeans.fit_predict(X_scaled)
)

print("\nCluster Distribution:")

print(
    df["behavior_cluster"]
    .value_counts()
    .sort_index()
)

# --------------------------------
# 16. Near-Failure Cluster Distribution
# --------------------------------

print("\nBehaviour Clusters Near Failure:")

near_failure_clusters = pd.crosstab(
    df["near_failure"],
    df["behavior_cluster"],
    normalize="index"
) * 100

print(
    near_failure_clusters.round(2)
)

# --------------------------------
# 17. Save Validation Results
# --------------------------------

output_path = (
    r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI"
    r"\results\model_validation_results.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nValidation results saved successfully!")

print(output_path)