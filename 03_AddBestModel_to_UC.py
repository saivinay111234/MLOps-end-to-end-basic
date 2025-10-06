# Databricks notebook source
# MAGIC %pip install --quiet mlflow --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC #### Find and Add best model from previous runs to Unity Catalog(UC)

# COMMAND ----------

model_name="sai_datastorage.default.mlops_churn"
current_user = spark.sql("SELECT current_user()").collect()[0][0]

# COMMAND ----------

import mlflow

xp_name = "mlops_customer_churn"
xp_path = f"/Users/{current_user}"
expriment_name = f"{xp_path}/{xp_name}"

print(f"Finding the best run from {xp_name} and pushing new model to {model_name}")
mlflow.set_experiment(f"{xp_path}/{xp_name}")

# COMMAND ----------

experiment_id = mlflow.search_experiments(filter_string=f"name LIKE '{xp_path}/{xp_name}%'", order_by=["last_update_time DESC"])[0].experiment_id
print(experiment_id)

# COMMAND ----------

best_model = mlflow.search_runs(experiment_ids=experiment_id, order_by=["metrics.val.f1_score DESC"], max_results=1, filter_string="status = 'FINISHED' and run_name='light_gbm_baseline'")
best_model

# COMMAND ----------

# Once we have our best model, we can now register it to the Unity Catalog Model Registry using its run ID
print(f"Registering model to {model_name}")
run_id = best_model.iloc[0]['run_id']

# Register the best model from experiments run to MLflow model registry
model_details = mlflow.register_model(f"runs:/{run_id}/sklearn_model", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Give the registered Model a description

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
# The main model description is typically done once.
client.update_registered_model(
  name=model_details.name,
  description="This model predicts whether a customer will churn using the features in the mlops_churn_training table. It is used to power the Telco Churn Dashboard in DB SQL."
)

# COMMAND ----------

# Add some more details on the new version we just registered.
best_score = best_model['metrics.val_f1_score'].values[0]
run_name = best_model['tags.mlflow.runName'].values[0]
version_desc = f"This model version has an F1 validation metric of {round(best_score,4)*100}%. Follow the link to its training run for more details."


client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description=version_desc
)


# We can also tag the model version with the F1 score for visibility
client.set_model_version_tag(
  name=model_details.name,
  version=model_details.version,
  key="f1_score",
  value=f"{round(best_score,4)}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set the latest model version as the Baseline/Challenger model

# COMMAND ----------

# MAGIC %md
# MAGIC We will set this newly registered model version as the Challenger (or Baseline) model. Challenger models are candidate models to replace the Champion model, which is the model currently in use.
# MAGIC
# MAGIC We will use the model's alias to indicate the stage it is at in its lifecycle.

# COMMAND ----------

# Set this version as the Challenger model, using its model alias
client.set_registered_model_alias(
  name=model_name,
  alias="Challenger", # Baseline
  version=model_details.version
)