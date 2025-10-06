# Databricks notebook source
# MAGIC %pip install --quiet lightgbm mlflow --upgrade
# MAGIC %restart_python

# COMMAND ----------

current_user = spark.sql("SELECT current_user()").collect()[0][0]
#Option 2: current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
current_user

# COMMAND ----------

import mlflow

xp_name = "mlops_customer_churn"
xp_path = f"/Users/{current_user}"

expriment_name = f"{xp_path}/{xp_name}"

try:
    experiment_id = mlflow.get_experiment_by_name(expriment_name).experiment_id
except Exception as e:
    print(f"Creating new experiment: {expriment_name}")
    experiment_id = mlflow.create_experiment(name=expriment_name, tags={"project": "customer_churn",
    "mlops_stage": "development"})

print(f"Experiment ID: {experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Data Lineage**
# MAGIC Data lineage is like version control for the training data.
# MAGIC - Model performance degrades - was it the code or the data?
# MAGIC - You need to reproduce a model exactly as it was trained
# MAGIC - Compliance/auditing requirements demand traceability
# MAGIC - Debugging "why did my model break?" becomes impossible without data tracking
# MAGIC
# MAGIC Creating data source object which reads metadata, reference, location etc.
# MAGIC > src_dataset = mlflow.data.load_delta(
# MAGIC >     table_name="sai_datastorage.default.mlops_churn_training"
# MAGIC > )
# MAGIC > display(src_dataset)
# MAGIC
# MAGIC Link the data to the mlflow run
# MAGIC > with mlflow.start_run():
# MAGIC >    .....
# MAGIC >    mlflow.log_input(src_dataset, context="training-input")
# MAGIC

# COMMAND ----------

latest_table_version = max(
    spark.sql(
        "describe history sai_datastorage.default.mlops_churn_training"
    ).toPandas()["version"]
)
print(f"Latest table version: {latest_table_version}")


src_dataset = mlflow.data.load_delta(table_name="sai_datastorage.default.mlops_churn_training", version=str(latest_table_version))

print(f"Source dataset: {src_dataset.name}")

# COMMAND ----------

df_loaded = src_dataset.df.filter("split='train'").drop("customerID","split")
display(df_loaded)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC **Boolean Columns**

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import FunctionTransformer


bool_imputers = []

bool_pipeline =Pipeline(steps=[
  ("cast_type", FunctionTransformer(lambda df:df.astype(object))),
  ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
  ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop='first'))
])

bool_transformers = [(
  "boolean", bool_pipeline
, ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"])]

bool_transformers

# COMMAND ----------

# MAGIC %md
# MAGIC **Numerical Columns**

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

num_imputers = []

num_imputers.append(("impute_mean", SimpleImputer(), ["tenure", "MonthlyCharges", "TotalCharges", "num_optional_services"]))

numerical_pipeline = Pipeline(steps=[
  ("converter", FunctionTransformer(lambda df:df.apply(pd.to_numeric, errors="coerce"))),
  ("imputers", ColumnTransformer(num_imputers)),
  ("standardizer", StandardScaler())
])

numerical_transformers = [("numerical", numerical_pipeline, ["tenure", "MonthlyCharges", "TotalCharges", "num_optional_services"])]

# COMMAND ----------

# MAGIC %md
# MAGIC **Categorical Columns**

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []
one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
])

categorical_one_hot_transformers = [
    ("onehot", one_hot_pipeline, ["Contract", "DeviceProtection", "InternetService", "MultipleLines", "OnlineBackup", "OnlineSecurity", "PaymentMethod", "StreamingMovies", "StreamingTV", "TechSupport"])
]


# COMMAND ----------

transformers = bool_transformers+numerical_transformers+categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
preprocessor

# COMMAND ----------

# MAGIC %md
# MAGIC **Train-Test-Split**

# COMMAND ----------

from sklearn.model_selection import train_test_split

label_col = "Churn"
X=df_loaded.toPandas()
X_train, X_val, Y_train, Y_val = train_test_split(X.drop(label_col, axis=1), X[label_col], test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Train the Classification Model**
# MAGIC - log relevant metrics to MLflow to track runs
# MAGIC - All runs are logged under an experiment in UI workspace
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment

# COMMAND ----------

import lightgbm

from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np


# COMMAND ----------

from mlflow.models import infer_signature, ModelSignature, Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline



def train_fn(params):
    with mlflow.start_run(experiment_id=experiment_id, run_name=params["run_name"]) as mlflow_run:
        lgbmclassifier = LGBMClassifier(**params)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", lgbmclassifier)
        ])
# Enable automatic logging of input samples, metrics, parameters and models
        mlflow.sklearn.autolog(log_models=False, silent=True)
        model.fit(X_train, Y_train)
        signature = infer_signature(X_train, Y_train) # infer model signature
        mlflow.sklearn.log_model(model, "sklearn_model", input_example=X_train.iloc[0].to_dict(), signature=signature)

        # Log training dataset object to capture upstream data lineage
        mlflow.log_input(src_dataset, context="training-input")

        # log metrics for the training set
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
        

        # Evaluates Models on training data
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(label_col):Y_train}),
            targets=label_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "training_" , "pos_label": "Yes" }
        )
        sklr_training_metrics = training_eval_result.metrics


        # Evaluates Models on validation data
        val_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_val.assign(**{str(label_col):Y_val}),
            targets=label_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "val_" , "pos_label": "Yes" }
        )
        sklr_val_metrics = val_eval_result.metrics

        loss = -sklr_val_metrics["val_f1_score"]

        # Truncate metric key names so they can be displayed together
        sklr_val_metrics = {k.replace("val_", ""): v for k, v in sklr_val_metrics.items()}

        return {
        "loss": loss,
        "val_metrics": sklr_val_metrics,
        "model": model,
        "run": mlflow_run,
        }

# COMMAND ----------

params = {
  "run_name": "light_gbm_baseline",
  "colsample_bytree": 0.4120544919020157, 
  "lambda_l1": 2.6616074270114995,
  "lambda_l2": 514.9224373768443,
  "learning_rate": 0.0678497372371143,
  "max_bin": 229,
  "max_depth": 8,
  "min_child_samples": 66,
  "n_estimators": 250,
  "num_leaves": 100,
  "path_smooth": 61.06596877554017,
  "subsample": 0.6965257092078714,
  "random_state": 42,
}

# COMMAND ----------

training_results = train_fn(params)

# COMMAND ----------

loss = training_results["loss"]
model = training_results["model"]
print(f"Model loss: {loss}")

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Confusion matrix, ROC, and Precision-Recall curves for validation data
# MAGIC ### We show the model's confusion matrix, RO,C and Precision-Recall curves on the validation data.**
# MAGIC
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

mlflow_run = training_results["run"]

# COMMAND ----------

# Click the link to see the MLflow run page
displayHTML(f"<a href=#mlflow/experiments/{mlflow_run.info.experiment_id}/runs/{ mlflow_run.info.run_id }/artifactPath/model> Link to model run page </a>")

# COMMAND ----------

import uuid
import os
from IPython.display import Image


# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve_plot.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve_plot.png")
display(Image(filename=eval_pr_curve_path))