from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_and_save_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/housing.csv", index=False)
    print("✅ Data saved to data/housing.csv")

if __name__ == "__main__":
    load_and_save_data()
