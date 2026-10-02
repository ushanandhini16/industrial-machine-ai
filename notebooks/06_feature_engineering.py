import pandas as pd

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

# Sort by machine and cycle
df = df.sort_values(["engine_id", "cycle"])

# Create behaviour features
for sensor in sensor_columns:

    # Change from previous cycle
    df[sensor + "_diff"] = df.groupby("engine_id")[sensor].diff()

    # Rolling mean
    df[sensor + "_rolling_mean"] = (
        df.groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )

    # Rolling standard deviation
    df[sensor + "_rolling_std"] = (
        df.groupby("engine_id")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).std())
    )

# Replace missing values created by diff/std
df = df.fillna(0)

print("\nOriginal sensor columns:")
print(sensor_columns)

print("\nNew feature columns:")
new_features = [
    col for col in df.columns
    if "_diff" in col or "_rolling_" in col
]
print(new_features)

print("\nOriginal shape after cleaning:")
print("Rows:", len(df))

print("\nNew shape after feature engineering:")
print(df.shape)

print("\nSample of new features:")
print(df[new_features].head(10))