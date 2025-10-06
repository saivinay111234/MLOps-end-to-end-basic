# Databricks notebook source
# MAGIC %pip install --quiet mlflow --upgrade
# MAGIC %restart_python

# COMMAND ----------

datapath='/Volumes/sai_datastorage/default/churnvolume/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = spark.read.csv(datapath, header=True, inferSchema=True)
display(df)

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("sai_datastorage.default.churn_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Preprocessing**

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

df = spark.read.table("sai_datastorage.default.churn_bronze")
df_pan = df.toPandas()
display(df_pan)

# COMMAND ----------

df_pan.isna().sum()

# COMMAND ----------

df_pan['TotalCharges']=pd.to_numeric(df_pan['TotalCharges'].replace(' ', np.nan), errors='coerce')


# COMMAND ----------

df_pan.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC **EDA**

# COMMAND ----------

# 1. Type of Internet service
df_pan['InternetService'].value_counts().plot(kind='pie', autopct='%1.1f%%')

# COMMAND ----------

# Data Balance
df_pan['Churn'].value_counts().plot(kind='bar', color=['yellow', 'green'])

# COMMAND ----------

# 3. Monthly charges vs Churn. 
#  This helps to see if higher paying customers churn more often or not
plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df_pan)
plt.title("Monthly Charges vs Churn")
plt.show()

# COMMAND ----------

plt.figure(figsize=(8,5))
sns.kdeplot(df_pan[df_pan['Churn']=="Yes"]['MonthlyCharges'], label="Churn=Yes", fill=True)
sns.kdeplot(df_pan[df_pan['Churn']=="No"]['MonthlyCharges'], label="Churn=No", fill=True)
plt.title("Distribution of Monthly Charges by Churn")
plt.legend()
plt.show()


# COMMAND ----------

plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="TotalCharges", data=df_pan)
plt.title("Monthly Charges vs Churn")
plt.show()

# COMMAND ----------

# Contract Type vs Churn.
# This helps to see if customers with long term contracts churn more often or not
plt.figure(figsize=(7,5))
sns.countplot(x="Contract", hue="Churn", data=df_pan)
plt.title("Contract Type vs Churn")
plt.show()


# COMMAND ----------

plt.figure(figsize=(8,5))
sns.kdeplot(df_pan[df_pan['Churn']=="Yes"]['PaperlessBilling'], label="Churn=Yes", fill=True)
sns.kdeplot(df_pan[df_pan['Churn']=="No"]['PaperlessBilling'], label="Churn=No", fill=True)
plt.title("Distribution of Monthly Charges by Churn")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Feature Engineering**

# COMMAND ----------

df_pan.head()

# COMMAND ----------

def feature_engineering(df):
    df_temp = df.copy()

    df_temp['SeniorCitizen'] = df_temp['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df_temp['num_optional_services'] = sum(map(lambda x: df_temp[x] == 'Yes', cols))

    return df_temp

# COMMAND ----------

df_clean = feature_engineering(df_pan)
df_clean.head()

# COMMAND ----------

#  Split the data into train and test sets
train_ratio, test_ratio = 0.8, 0.2

# Add a random column with fixed seed for reproducibility
np.random.seed(42)
df_clean["random"] = np.random.rand(len(df_clean))

# Assign split column based on threshold
df_clean["split"] = np.where(df_clean["random"] < train_ratio, "train", "test")

# Drop helper column if you don't need it
df_clean = df_clean.drop(columns=["random"])

# Verify counts
print(df_clean["split"].value_counts(normalize=True))

# COMMAND ----------

# Write to data table
churn_features_spark = spark.createDataFrame(df_clean)
(churn_features_spark.write.mode("overwrite")
               .option("overwriteSchema", "true")
               .saveAsTable("sai_datastorage.default.mlops_churn_training"))

catalog = "sai_datastorage"
db = "default"

spark.sql(f"""
COMMENT ON TABLE {catalog}.{db}.mlops_churn_training IS 
'The features in this table are derived from the churn_bronze table in the lakehouse. 
We created service features and cleaned up their names. No aggregations were performed.'
""")

