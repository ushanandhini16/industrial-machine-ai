import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
# 5. Feature Selection
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

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_columns
)

# --------------------------------
# 7. K-Means Clustering
# --------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["behavior_cluster"] = (
    kmeans.fit_predict(X_scaled)
)

# --------------------------------
# 8. Cluster Centroids
# --------------------------------

centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=feature_columns
)

print("\nCluster Centroid Shape:")
print(centroids.shape)

# --------------------------------
# 9. Compare Cluster 1 and 3
# --------------------------------

comparison = (
    centroids.loc[[1, 3]]
    .T
)

comparison.columns = [
    "Cluster_1",
    "Cluster_3"
]

comparison["absolute_difference"] = (
    comparison["Cluster_1"]
    - comparison["Cluster_3"]
).abs()

comparison = comparison.sort_values(
    "absolute_difference",
    ascending=False
)

print("\nTop Features Distinguishing Cluster 1 and Cluster 3:")

print(
    comparison.head(20).round(3)
)

# --------------------------------
# 10. Cluster Statistics
# --------------------------------

cluster_means = (
    X_scaled_df
    .assign(cluster=df["behavior_cluster"])
    .groupby("cluster")
    .mean()
)

print("\nTop Distinguishing Features Across Clusters:")

# Calculate variation between clusters
feature_variation = (
    cluster_means.max()
    - cluster_means.min()
)

top_features = (
    feature_variation
    .sort_values(ascending=False)
    .head(20)
)

print(
    top_features.round(3)
)

# --------------------------------
# 11. Original Sensor Comparison
# --------------------------------

original_sensor_means = (
    df.groupby("behavior_cluster")[sensor_columns]
    .mean()
)

print("\nOriginal Sensor Means by Cluster:")

print(
    original_sensor_means.round(3)
)

# --------------------------------
# 12. Near-Failure Analysis
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

df["life_progress"] = (
    df["cycle"] / df["max_cycle"]
)

df["near_failure"] = (
    df["life_progress"] >= 0.80
).astype(int)

near_failure_counts = pd.crosstab(
    df["near_failure"],
    df["behavior_cluster"],
    normalize="index"
) * 100

print("\nBehaviour Distribution:")

print(
    near_failure_counts.round(2)
)

# --------------------------------
# 13. Most Important Features
# --------------------------------

print("\nTop 10 AI-Discovered Features:")

for feature in top_features.head(10).index:

    print(
        feature,
        "->",
        round(top_features[feature], 3)
    )

# --------------------------------
# 14. Save Explanation Results
# --------------------------------

output_path = (
    r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI"
    r"\results\ai_behavior_explanation.csv"
)

comparison.to_csv(
    output_path
)

print("\nAI explanation results saved successfully!")

print(output_path)