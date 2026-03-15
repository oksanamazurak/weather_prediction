import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder

input_file = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)
df = df.dropna(subset=["RainTomorrow"])
df["RainTomorrow"] = df["RainTomorrow"].map({"Yes":1, "No":0})
df["RainToday"] = df["RainToday"].map({"Yes":1, "No":0})

numeric_cols = df.select_dtypes(include=["float64","int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df["Temp_diff"] = df["MaxTemp"] - df["MinTemp"]

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
print(f"Prepared data saved to {output_dir}")