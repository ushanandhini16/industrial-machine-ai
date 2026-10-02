import pandas as pd
import matplotlib.pyplot as plt
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

# Load data
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
df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

# Feature engineering
for sensor in sensor_columns:

    df[sensor + "_diff"] = (
        df.groupby("engine_id")[sensor].diff()
    )

    df[sensor + "_rolling_mean"] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    )

    df[sensor + "_rolling_std"] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
    )

df = df.fillna(0)

# ML features
feature_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_columns]

# Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(X)

df["anomaly"] = model.predict(X)
df["anomaly_score"] = model.decision_function(X)

df["anomaly_label"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# ---------------------------------------
# 1. Anomaly count for each engine
# ---------------------------------------

anomaly_counts = (
    df[df["anomaly_label"] == "Anomaly"]
    .groupby("engine_id")
    .size()
    .sort_values(ascending=False)
)

print("\nAnomaly Count by Engine:")
print(anomaly_counts.head(15))

# ---------------------------------------
# 2. Anomaly percentage by engine
# ---------------------------------------

engine_total = df.groupby("engine_id").size()

anomaly_percentage = (
    anomaly_counts / engine_total * 100
).sort_values(ascending=False)

print("\nAnomaly Percentage by Engine:")
print(anomaly_percentage.head(15).round(2))

# ---------------------------------------
# 3. Most unusual observations
# ---------------------------------------

most_unusual = df.sort_values("anomaly_score").head(15)

print("\nMost Unusual Observations:")
print(
    most_unusual[
        ["engine_id", "cycle", "anomaly_score", "anomaly_label"]
    ]
)

# ---------------------------------------
# 4. Plot anomaly timeline for Engine 1
# ---------------------------------------

engine_id = 1

engine_data = df[df["engine_id"] == engine_id]

plt.figure(figsize=(12, 5))

plt.plot(
    engine_data["cycle"],
    engine_data["anomaly_score"],
    label="Anomaly Score"
)

anomalies = engine_data[
    engine_data["anomaly_label"] == "Anomaly"
]

plt.scatter(
    anomalies["cycle"],
    anomalies["anomaly_score"],
    label="Detected Anomaly"
)

plt.axhline(
    0,
    linestyle="--",
    label="Anomaly Boundary"
)

plt.xlabel("Cycle")
plt.ylabel("Anomaly Score")
plt.title("Engine 1 - Anomaly Timeline")
plt.legend()
plt.tight_layout()
plt.show()