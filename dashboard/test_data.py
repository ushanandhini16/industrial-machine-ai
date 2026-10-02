import pandas as pd

file_path = r"C:\Users\ushan\OneDrive\Desktop\Industrial_Machine_AI\results\failure_association_results.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nAnomaly Count:")
print(df["anomaly"].value_counts())

print("\nBehaviour Cluster Count:")
print(df["behavior_cluster"].value_counts().sort_index())

print("\nLife Stage Count:")
print(df["life_stage"].value_counts())