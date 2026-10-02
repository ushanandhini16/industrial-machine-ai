import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

print("Original feature shape:", X.shape)

# Standardization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaled feature shape:", X_scaled.shape)

# K-Means clustering
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["behavior_cluster"] = kmeans.fit_predict(X_scaled)

# Cluster counts
print("\nBehavior Cluster Counts:")
print(
    df["behavior_cluster"]
    .value_counts()
    .sort_index()
)

# Cluster percentage
print("\nBehavior Cluster Percentage:")
print(
    (
        df["behavior_cluster"]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2)
)

# Show sample
print("\nSample Behavior Clusters:")
print(
    df[
        [
            "engine_id",
            "cycle",
            "behavior_cluster"
        ]
    ].head(20)
)
