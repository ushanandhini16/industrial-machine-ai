import pandas as pd

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

# Select sensor columns
sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor")
]

# Calculate statistics
analysis = df[sensor_columns].describe().T

print("\nSensor Analysis:")
print(analysis[["mean", "std", "min", "max"]])

# Sort sensors based on variation
print("\nSensors sorted by variation:")
print(
    analysis[["std"]]
    .sort_values(by="std", ascending=False)
)