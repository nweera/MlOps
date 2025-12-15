import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Drop missing values
if config["preprocessing"]["drop_missing"]:
    df = df.dropna()

# Handle Iris target column
if "Species" in df.columns:
    df["target"] = df["Species"].astype("category").cat.codes
    df = df.drop("Species", axis=1)

X = df.drop("target", axis=1)
y = df["target"]

# Scaling
method = config["preprocessing"]["method"]

if method == "standard_scaler":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

elif method == "minmax":
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["data"]["test_size"],
    shuffle=config["data"]["shuffle"],
    random_state=config["training"]["random_state"]
)

# Save processed data
os.makedirs("data/processed", exist_ok=True)

pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Data preparation completed.")
