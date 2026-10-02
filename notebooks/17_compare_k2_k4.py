import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI\results\failure_association_results.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print("Rows:", len(df))

# --------------------------------------------------
# 2. Original 60 ML features
# --------------------------------------------------

sensor_columns = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

feature_columns = []

for sensor in sensor_columns:
    feature_columns.append(sensor)
    feature_columns.append(sensor + "_diff")
    feature_columns.append(sensor + "_rolling_mean")
    feature_columns.append(sensor + "_rolling_std")

X = df[feature_columns].copy()

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# --------------------------------------------------
# 3. Standardization
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 4. Compare k=2 and k=4
# --------------------------------------------------

comparison = []

for k in [2, 4]:

    print("\n" + "=" * 50)
    print("K =", k)
    print("=" * 50)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    # Silhouette
    silhouette = silhouette_score(
        X_scaled,
        labels,
        sample_size=min(5000, len(X_scaled)),
        random_state=42
    )

    print("Silhouette Score:", round(silhouette, 4))

    # --------------------------------------------------
    # Cluster size
    # --------------------------------------------------

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    print("\nCluster Sizes:")

    for cluster, count in cluster_counts.items():

        percentage = (count / len(df)) * 100

        print(
            f"Cluster {cluster}: "
            f"{count} observations "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------
    # Add temporary cluster labels
    # --------------------------------------------------

    temp_df = df.copy()

    temp_df["comparison_cluster"] = labels

    # --------------------------------------------------
    # Anomaly concentration
    # --------------------------------------------------

    print("\nAnomaly Concentration:")

    for cluster in sorted(temp_df["comparison_cluster"].unique()):

        cluster_data = temp_df[
            temp_df["comparison_cluster"] == cluster
        ]

        anomaly_count = cluster_data["anomaly"].sum()

        anomaly_rate = (
            anomaly_count / len(cluster_data)
        ) * 100

        print(
            f"Cluster {cluster}: "
            f"{anomaly_count} anomalies "
            f"({anomaly_rate:.2f}%)"
        )

    # --------------------------------------------------
    # Life-stage distribution
    # --------------------------------------------------

    print("\nLife Stage Distribution:")

    life_table = pd.crosstab(
        temp_df["comparison_cluster"],
        temp_df["life_stage"],
        normalize="index"
    ) * 100

    print(life_table.round(2))

    # --------------------------------------------------
    # Near-failure association
    # --------------------------------------------------

    temp_df["near_failure"] = (
        temp_df["life_progress"] >= 0.80
    )

    print("\nNear-Failure Distribution:")

    near_failure_table = pd.crosstab(
        temp_df["comparison_cluster"],
        temp_df["near_failure"],
        normalize="index"
    ) * 100

    print(near_failure_table.round(2))

    # --------------------------------------------------
    # Save comparison metrics
    # --------------------------------------------------

    comparison.append({
        "k": k,
        "silhouette_score": silhouette,
        "clusters": k,
        "inertia": model.inertia_
    })


# --------------------------------------------------
# 5. Save summary
# --------------------------------------------------

comparison_df = pd.DataFrame(comparison)

output_file = r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI\results\k2_k4_comparison.csv"

comparison_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 50)
print("Comparison Summary")
print("=" * 50)

print(comparison_df.round(4))

print("\nComparison saved to:")
print(output_file)

print("\nStep 21 completed successfully.")