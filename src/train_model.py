import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

X_train, X_test, y_train, y_test = joblib.load("data/processed/housing_processed.pkl")

mlflow.set_experiment("housing_price_prediction")

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        
        model.fit(X_train, y_train)

        
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_name", model_name)
        if model_name == "DecisionTree":
            mlflow.log_param("max_depth", 5)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

    
        mlflow.sklearn.log_model(model, "model")

        print(f"Trained and logged {model_name} with MSE={mse:.2f}, R2={r2:.2f}")
