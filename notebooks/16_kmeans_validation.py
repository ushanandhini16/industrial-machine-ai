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
# 2. Select the ORIGINAL 60 ML features
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

print("\nNumber of ML features:", len(feature_columns))

# --------------------------------------------------
# 3. Check missing columns
# --------------------------------------------------

missing_columns = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_columns:

    print("\nMissing columns:")
    print(missing_columns)

    raise ValueError(
        "Some original ML feature columns are missing."
    )

# --------------------------------------------------
# 4. Create feature matrix
# --------------------------------------------------

X = df[feature_columns].copy()

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Replace missing values
X = X.fillna(0)

# --------------------------------------------------
# 5. Standardization
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed")

# --------------------------------------------------
# 6. KMeans validation
# --------------------------------------------------

k_values = range(2, 9)

inertia_values = []
silhouette_values = []

print("\nKMeans Validation Results")
print("-" * 55)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels,
        sample_size=min(5000, len(X_scaled)),
        random_state=42
    )

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)

    print(
        f"k={k} | "
        f"Inertia={inertia:.2f} | "
        f"Silhouette={silhouette:.4f}"
    )

# --------------------------------------------------
# 7. Save validation results
# --------------------------------------------------

validation_results = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertia_values,
    "silhouette_score": silhouette_values
})

output_file = r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI\results\kmeans_validation_results.csv"

validation_results.to_csv(
    output_file,
    index=False
)

print("\nValidation results saved to:")
print(output_file)

# --------------------------------------------------
# 8. Best silhouette score
# --------------------------------------------------

best_row = validation_results.loc[
    validation_results["silhouette_score"].idxmax()
]

print("\nHighest Silhouette Score:")
print("k =", int(best_row["k"]))
print("Score =", round(best_row["silhouette_score"], 4))

# --------------------------------------------------
# 9. Existing k=4 score
# --------------------------------------------------

k4_row = validation_results[
    validation_results["k"] == 4
].iloc[0]

print("\nExisting Project Choice (k=4):")
print("Silhouette Score =", round(
    k4_row["silhouette_score"], 4
))

print("\nValidation completed successfully.")