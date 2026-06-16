# Model Registryو From Experiment to Production
# Stage transitions: None -> Staging -> Production -> Archived


import mlflow
import mlflow.pyfunc
import time
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_experiment("07-production")
client = MlflowClient()

# Create a simple model and register it

class SimpleLLMWrapper(mlflow.pyfunc.PythonModel):
    """A simple wrapper to demonstrate the Model Registry workflow"""

    def predict(self, context, model_input):
        questions = model_input["question"].tolist()
        return [f"Answer to: {q}" for q in questions]


MODEL_NAME = "production-llm-v1"

with mlflow.start_run(run_name="register-for-production"):

    mlflow.log_param("model_type", "LLM Wrapper")
    mlflow.log_metric("eval_score", 0.87)

    mlflow.pyfunc.log_model(
        python_model=SimpleLLMWrapper(),
        artifact_path="llm-model",
        registered_model_name=MODEL_NAME,
    )

    run_id = mlflow.active_run().info.run_id

print(f"Model registered. Run ID: {run_id}")

# Move model through lifecycle stages
time.sleep(2)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = versions[0].version

print(f"\nModel version: {latest_version}")
print("  Current stage: None (just registered)")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Staging",
)
print("  Moved to: Staging")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Production",
)
print("  Moved to: Production")

# Load the model from Production (as in deployment)
print("\nLoading model from Production...")

production_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/Production"
)

test_input = pd.DataFrame({"question": ["What is MLflow?", "What does LLMOps mean?"]})
predictions = production_model.predict(test_input)

print("\nProduction model output:")
for q, p in zip(test_input["question"], predictions):
    print(f"  Q: {q}")
    print(f"  A: {p}\n")

print("This is the exact workflow every LLMOps Engineer follows.")
print("Go to MLflow UI -> Models -> production-llm-v1")
