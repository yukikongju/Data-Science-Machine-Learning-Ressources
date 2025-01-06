import mlflow 
from mlflow import MlflowClient

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # ---- MLFlow Metadata Setup ----
    HOST, PORT = "127.0.0.1", 8080
    mlflow.set_tracking_uri(uri=f"http://{HOST}:{PORT}")
    mlflow.set_experiment("MLflow Quickstart")


    # --- load iris dataset
    X, y = datasets. load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=420)

    # --- load model from registry
    client = MlflowClient()
    MODEL_NAME, ARTIFACT_PATH = "tracking-quickstart", "iris_model"
    MODEL_VERSION_ALIAS = "test_model"
    client.set_registered_model_alias(MODEL_NAME, MODEL_VERSION_ALIAS, "1") # from MLFLOWUI

    model_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_VERSION_ALIAS)
    model_tags = model_info.tags
    print(model_tags)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_VERSION_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri) 
    print(model)

    # --- make predictions 
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")



if __name__ == "__main__":
    main()
