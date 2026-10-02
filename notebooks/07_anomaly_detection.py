import pandas as pd
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

# Sort data
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

# Remove missing values
df = df.fillna(0)

# Select ML features
feature_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_columns]

print("Number of ML features:", len(feature_columns))
print("Data shape:", X.shape)

# Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

# Train model
model.fit(X)

# Prediction
df["anomaly"] = model.predict(X)

# Anomaly score
df["anomaly_score"] = model.decision_function(X)

# Convert prediction
df["anomaly_label"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# Results
print("\nAnomaly Count:")
print(df["anomaly_label"].value_counts())

print("\nAnomaly Percentage:")
print(
    round(
        (df["anomaly_label"] == "Anomaly").mean() * 100,
        2
    ),
    "%"
)

print("\nSample Anomalies:")
print(
    df[df["anomaly_label"] == "Anomaly"][
        ["engine_id", "cycle", "anomaly_score", "anomaly_label"]
    ].head(10)
)