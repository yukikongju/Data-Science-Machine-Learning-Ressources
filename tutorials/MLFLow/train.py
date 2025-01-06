# Tutorial: https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # ---- Train & Predict ----
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=420)

    #  https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
    params = {"solver": "lbfgs", "max_iter": 100, 
              "penalty": "l2", "multi_class": "auto", "random_state": 420}

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='positive', average='micro')
    recall = recall_score(y_test, y_pred, pos_label='positive', average='micro')
    f1 = f1_score(y_test, y_pred, pos_label='positive', average='micro')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # ---- MLFlow Metadata Setup ----
    HOST, PORT = "127.0.0.1", 8080
    mlflow.set_tracking_uri(uri=f"http://{HOST}:{PORT}")
    mlflow.set_experiment("MLflow Quickstart")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # infer model signature
        mlflow.set_tag("Training Info", "Basic Logistic Regression for Iris model")
        signature = infer_signature(X_train, model.predict(X_test))

        # log the model
        model_info = mlflow.sklearn.log_model(
                sk_model=model, artifact_path="iris_model", signature=signature, 
                input_example=X_train, registered_model_name="tracking-quickstart"
                )

        # ---- Load the model and use it for inference ----
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(X_test)

        iris_feature_names = datasets.load_iris().feature_names

        results = pd.DataFrame(X_test, columns=iris_feature_names)
        results["actual"] = y_test
        results["predicted"] = predictions

        results.to_csv("output.csv")








if __name__ == "__main__":
    main()

