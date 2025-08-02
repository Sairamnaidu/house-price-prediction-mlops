import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


df = pd.read_csv("../data/housing.csv")


X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_preprocessed = preprocessor.fit_transform(X)


os.makedirs("../models", exist_ok=True)

joblib.dump(preprocessor, "../models/preprocessor.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

os.makedirs("../data/processed", exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), "../data/processed/housing_processed.pkl")

print("Preprocessing completed. Files saved in ../models/ and ../data/processed/")
