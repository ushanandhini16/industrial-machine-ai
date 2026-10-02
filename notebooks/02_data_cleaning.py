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

print("Original shape:", df.shape)

# Find columns having only one unique value
constant_columns = [
    col for col in df.columns
    if df[col].nunique() <= 1
]

print("\nConstant columns:")
print(constant_columns)

# Remove constant columns
df = df.drop(columns=constant_columns)

print("\nShape after removing constant columns:", df.shape)

print("\nRemaining columns:")
print(df.columns.tolist())