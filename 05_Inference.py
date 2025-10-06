# Databricks notebook source
# MAGIC %pip install --quiet mlflow --upgrade
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying the model for batch inferences
# MAGIC Now that our model is available in the Unity Catalog Model Registry, we can load it to compute our inferences and save them in a table to start building dashboards.
# MAGIC
# MAGIC We will use the MLflow function to load a pyspark UDF and distribute our inference in the entire cluster. We can load the model with plain Python and use a Pandas Dataframe if the data is small.

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

req_path = ModelsArtifactRepository(f"models:/sai_datastorage.default.mlops_churn@Champion").download_artifacts(artifact_path="requirements.txt")
print(open(req_path).read())

# COMMAND ----------

# MAGIC %pip install --quiet -r $req_path
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference in Champion Model
# MAGIC We are ready to run inference on the Champion model. We will load the model as a Spark UDF and generate predictions for our customer records.

# COMMAND ----------

test_df = spark.table("sai_datastorage.default.mlops_churn_training")
# display(test_df)
champion_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/sai_datastorage.default.mlops_churn@Champion")
preds_df = test_df.withColumn('predictions', champion_model(*champion_model.metadata.get_input_schema().input_names()))

display(preds_df)