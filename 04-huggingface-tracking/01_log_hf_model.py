import mlflow
import mlflow.pyfunc
import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("AVVALAI_API_KEY")
if not API_KEY:
    raise ValueError("AVVALAI_API_KEY is not set in .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.avalai.ir/v1"
)

mlflow.set_experiment("04-avvalai-tracking")

MODEL_NAME = "gpt-4o-mini"

# AvvalAI generation function
def generate(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Answer briefly and clearly."},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


# MLflow PyFunc Wrapper
class AvvalAIWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from openai import OpenAI
        import os

        self.client = OpenAI(
            api_key=os.getenv("AVVALAI_API_KEY"),
            base_url="https://api.avalai.ir/v1"
        )
        self.model = "gpt-4o-mini"

    def predict(self, context, model_input):
        questions = model_input["question"].tolist()

        outputs = []
        for q in questions:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Answer briefly and clearly."},
                    {"role": "user", "content": q}
                ],
                temperature=0.7
            )
            outputs.append(res.choices[0].message.content)

        return outputs


# Test data
test_questions = [
    "What is machine learning?",
    "Explain MLflow in one sentence.",
    "What is the difference between training and inference?"
]

with mlflow.start_run(run_name="avvalai-gpt4o-mini-eval"):

    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("provider", "AvvalAI")
    mlflow.log_param("framework", "API")

    latencies = []

    for i, q in enumerate(test_questions):

        start = time.time()
        output = generate(q)
        latency = (time.time() - start) * 1000

        latencies.append(latency)

        print(f"\nQ: {q}")
        print(f"A: {output}")

        mlflow.log_metric("latency_ms", latency, step=i)

    mlflow.log_metric("avg_latency_ms", sum(latencies) / len(latencies))

    # Log model in MLflow
    mlflow.pyfunc.log_model(
        artifact_path="avvalai-model",
        python_model=AvvalAIWrapper()
    )

print("\nAvvalAI model logged to MLflow successfully")