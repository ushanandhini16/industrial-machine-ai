import pandas as pd
import matplotlib.pyplot as plt

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

# Calculate correlation
correlation = df[sensor_columns].corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))

# Plot correlation matrix
plt.figure(figsize=(12, 9))

plt.imshow(correlation, cmap="coolwarm", aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(sensor_columns)),
    sensor_columns,
    rotation=90
)

plt.yticks(
    range(len(sensor_columns)),
    sensor_columns
)

plt.title("Sensor Correlation Matrix")

plt.tight_layout()
plt.show()