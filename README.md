# MLOps-end-to-end-basic
Built MLOps pipeline end-to-end from data ingestion, cleaning, feature engineering, training, validating, selecting best model, model validation, promotion and inference.

1. Data Ingestion, Prep & Feature Engineering ⚙️
The Grind: This is where the magic (and cleaning) happens! I started with raw data, tackled missing values in TotalCharges, and engineered key features like num_optional_services.

Tech Stack: I used Spark and Pandas to transform the data, saving the clean, split data as a Silver Delta Table (mlops_churn_training).

⬇️

2. Experimentation & Model Training 🧪
The Brains: I set up an MLflow Experiment to track everything. The model (a LightGBM classifier) was built inside a scikit-learn Pipeline, and I made sure to log Data Lineage so I always know exactly which data version trained my model.

Key Output: The trained model artifact and crucial performance metrics like F1, ROC, and PR curves.

⬇️

3. Model Registration & Staging 📦
The Promotion: Found the best performing run and officially Registered it to the Unity Catalog (UC) Model Registry—a single source of truth for my models.

Staging: The model started its journey with the Challenger alias, awaiting its final test. This is essential for proper version control and governance!

⬇️

4. Validation & Promotion Gate ✅
The Gatekeeper: No model gets promoted on my watch without passing strict rules! I implemented checks for governance (e.g., must have a detailed description) and Performance Validation (Challenger F1 must be ≥ Champion F1).

Business Check: I even simulated the Business Metrics impact to see if the model adds tangible revenue value. Only after passing did it earn the Champion alias!

⬇️

5. Production Deployment & Inference 🚀
The Launch: Time to score! I loaded the model using its production-ready @Champion alias, guaranteeing I always get the latest, validated version.

Execution: Leveraging Databricks, I used an mlflow.pyfunc.spark_udf for highly scalable, distributed Batch Scoring, spitting out predictions for every customer.

What's Next? Scaling Up!
This project is the foundational, most basic MLOps setup—a great start for any production workload. But the learning doesn't stop here!
