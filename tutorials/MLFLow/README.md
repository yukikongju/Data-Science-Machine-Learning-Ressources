# MLFlow 

What we learned:
- How to start an MLflow tracking server with `mlflow.set_tracking_uri` or `MLflowClient`
- How to store a model params, metrics, tag with `mlflow.log_params()`, 
  `mlflow.log_metric(<metric>, <value>)`, `mlflow.set_tag()`, 
  `mlflow.<MODEL_FLAVOR>.log_model`
- How to load a model using:
    * model_info.model_uri (in train.py)
    * client model_uri (in test.py)
- 


To learn:
- How to setup an experiment and lifecycle with `client.create_experiment`
    * [tut](https://mlflow.org/docs/latest/getting-started/logging-first-model/step3-create-experiment.html)


Tutorials:
- [X] [MLfLow Model Registry](https://mlflow.org/docs/latest/getting-started/registering-first-model/index.html)
    - [ ] [MLFlow Model Registry Docs](https://mlflow.org/docs/latest/model-registry.html)
- [ ] [MLFlow for ARIMA/statsmodel](https://mlflow.org/docs/latest/models.html#statsmodels-statsmodels)
- [ ] [MLFlow for Torch](https://mlflow.org/docs/latest/deep-learning/pytorch/guide/index.html)
- [ ] [MLFlow Deployment](https://mlflow.org/docs/latest/deployment/index.html)
- [ ] [MLFlow Recipes](https://mlflow.org/docs/latest/recipes.html)



TODO:
- [in-depth tutorial](https://mlflow.org/docs/latest/getting-started/logging-first-model/index.html)
- [pytorch tutorial](https://mlflow.org/docs/latest/deep-learning/pytorch/guide/index.html)
- [hyperparameter tuning w/ MLFlow and Optuna](https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html)
- [MLFLow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing)
- [MLFlow New Features](https://mlflow.org/docs/latest/new-features/index.html)


