# Databricks notebook source
# MAGIC %pip install --quiet mlflow --upgrade

# COMMAND ----------

import mlflow

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

req_path = ModelsArtifactRepository(
    f"models:/sai_datastorage.default.mlops_churn@Challenger"
).download_artifacts(artifact_path="requirements.txt")

# COMMAND ----------

# Now Install all libraries from model req_path
%pip install --quiet -r $req_path
%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch the Model Information

# COMMAND ----------

# We will fetch the model information for Challenger model from Unity Catalog

# we are interested in validating "Challenger" model

from mlflow import MlflowClient
model_alias = "Challenger"
model_name = "sai_datastorage.default.mlops_churn"

client = MlflowClient()
model_details = client.get_model_version_by_alias(model_name, model_alias)
model_version = int(model_details.version)

print(f"Validating {model_alias} model for {model_name} on model version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Validation - Section 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Description Checks
# MAGIC Has the data scientist provided a model description being submitted?

# COMMAND ----------

# If there's no description or an insufficient number of characters, tag accordingly
if not model_details.description:
  has_description = False
  print("Please add model description")
elif not len(model_details.description) > 20:
  has_description = False
  print("Please add detailed model description (40 char min).")
else:
  has_description = True

print(f'Model {model_name} version {model_details.version} has description: {has_description}')
client.set_model_version_tag(name=model_name, version=str(model_details.version), key="has_description", value=has_description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Performance metric
# MAGIC
# MAGIC We want to validate the model performance metric. Typically, we want to compare this metric obtained for the Challenger model against that of the Champion model. Since we have yet to register a Champion model, we will only retrieve the metric for the Challenger model without doing a comparison.
# MAGIC
# MAGIC The registered model captures information about the MLflow experiment run, where the model metrics were logged during training. This gives you traceability from the deployed model back to the initial training runs.
# MAGIC
# MAGIC Here, we will use the F1 score for the out-of-sample test data set aside at training time.

# COMMAND ----------

import mlflow

model_run_id = model_details.run_id
f1_score = mlflow.get_run(model_run_id).data.metrics["val_f1_score"]

try:
    # Compare the challenger f1 score to the existing champion if it exists
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
    champion_f1 = mlflow.get_run(champion_model.run_id).data.metrics["val_f1_score"]
    print(f'Champion f1 score: {champion_f1}. Challenger f1 score: {f1_score}.')
    metric_f1_passed = f1_score>=champion_f1
except:
    print(f"No Champion found. Accept the model as it's the first one.")
    metric_f1_passed = True

print(f'Model {model_name} version {model_details.version} metric_f1_passed: {metric_f1_passed}')
# Tag that F1 metric check has passed
client.set_model_version_tag(name=model_name, version=model_details.version, key="metric_f1_passed", value=metric_f1_passed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Benchmark or business metrics on the eval dataset
# MAGIC
# MAGIC Let's use our validation dataset to check the potential new model impact.
# MAGIC
# MAGIC Note: This is just to evaluate our models, not to be confused with A/B testing. A/B testing is done online, splitting the traffic between 2 models. It requires a feedback loop to evaluate the effect of the prediction (e.g., after a prediction, did the discount we offered to the customer prevent the churn?)

# COMMAND ----------

import pyspark.sql.functions as F

eval_df = spark.table('mlops_churn_training').filter("split='test'")
# display(eval_df)


# Call the model with the given alias and return the prediction
def predict_churn(df, model_alias):
  model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/sai_datastorage.default.mlops_churn@{model_alias}")

  return df.withColumn('predictions', model(*model.metadata.get_input_schema().input_names()))

# COMMAND ----------

import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix


#Note: this is over-simplified and depends on your use-case, but the idea is to evaluate our model against business metrics
cost_of_customer_churn = 2000 #in dollar
cost_of_discount = 500 #in dollar

cost_true_negative = 0 #did not churn, we did not give him the discount
cost_false_negative = cost_of_customer_churn #did churn, we lost the customer
cost_true_positive = cost_of_customer_churn -cost_of_discount #We avoided churn with the discount
cost_false_positive = -cost_of_discount #doesn't churn, we gave the discount for free

def get_model_value_in_dollar(model_alias):
    # Convert preds_df to Pandas DataFrame
    model_predictions = predict_churn(eval_df, model_alias).toPandas()
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(model_predictions['Churn'], model_predictions['predictions']).ravel()
    return tn * cost_true_negative+ fp * cost_false_positive + fn * cost_false_negative + tp * cost_true_positive
#add an exception to catch non-existing model champion yet
is_champ_model_exist = True
try:
    client.get_model_version_by_alias(f"sai_datastorage.default.mlops_churn", "Champion")
    print("Model already registered as Champion")
except Exception as error:
    print("An error occurred:", type(error).__name__, "It means no champion model yet exist")
    is_champ_model_exist = False
if is_champ_model_exist:
    champion_potential_revenue_gain = get_model_value_in_dollar("Champion")
    challenger_potential_revenue_gain = get_model_value_in_dollar("Challenger")

try:
    #Compare the challenger f1 score to the existing champion if it exists
    champion_potential_revenue_gain = get_model_value_in_dollar("Champion")
except:
    print(f"No Champion found. Accept the model as it's the first one.")
    champion_potential_revenue_gain = 0
    
challenger_potential_revenue_gain = get_model_value_in_dollar("Challenger")

data = {'Model Alias': ['Challenger', 'Champion'],
        'Potential Revenue Gain': [challenger_potential_revenue_gain, champion_potential_revenue_gain]}

# Create a bar plot using plotly express
px.bar(data, x='Model Alias', y='Potential Revenue Gain', color='Model Alias',
    labels={'Potential Revenue Gain': 'Revenue Impacted'},
    title='Business Metrics - Revenue Impacted')

# COMMAND ----------

# Validation results
results = client.get_model_version(model_name, model_version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC # Promoting the Challenger to Champion
# MAGIC When we are satisfied with the results of the Challenger model, we can promote it to Champion. This is done by setting its alias to @Champion. Inference pipelines that load the model using the @Champion alias will then load this new model. The alias on the older Champion model, if there is one, will be automatically unset. The model retains its @Challenger alias until a newer Challenger model is deployed with the alias to replace it.

# COMMAND ----------

if results.tags["has_description"] == "True" and results.tags["metric_f1_passed"] == "True":
  print('register model as Champion!')
  client.set_registered_model_alias(
    name=model_name,
    alias="Champion",
    version=model_version
  )
else:
  raise Exception("Model not ready for promotion")