import pandas as pd

df = pd.read_csv("data/raw/private_5g_iot_dataset_final.csv")
print(df.head())
print(df.shape)
print(df.columns.tolist())
