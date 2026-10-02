import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PCA_1"] = X_pca[:, 0]
df["PCA_2"] = X_pca[:, 1]

print("\nExplained Variance:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(
    round(
        pca.explained_variance_ratio_.sum() * 100,
        2
    ),
    "%"
)

# Plot
plt.figure(figsize=(12, 8))

for cluster in sorted(df["behavior_cluster"].unique()):

    cluster_data = df[
        df["behavior_cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["PCA_1"],
        cluster_data["PCA_2"],
        label=f"Cluster {cluster}",
        alpha=0.5,
        s=15
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("AI-Discovered Machine Behaviour Map")
plt.legend()
plt.tight_layout()
plt.show()