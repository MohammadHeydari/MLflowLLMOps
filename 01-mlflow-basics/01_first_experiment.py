# First MLflow Experiment (Real LLM API)

import os
import mlflow
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

API_KEY = os.getenv("AVVALAI_API_KEY")
if not API_KEY:
    raise ValueError("AVVALAI_API_KEY is not set in .env")

BASE_URL = "https://api.avalai.ir/v1"

# Create OpenAI-compatible client (AvvalAI)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# MLflow setup
mlflow.set_experiment("01-llm-basics")

# Real LLM call
def llm_call(prompt: str, model: str, temperature: float) -> dict:
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    end = time.time()

    output = response.choices[0].message.content

    return {
        "output": output,
        "tokens_used": getattr(response.usage, "total_tokens", 0),
        "latency_ms": int((end - start) * 1000),
    }


# Experiment
prompt = "Explain the difference between Machine Learning and Deep Learning"
model_name = "gpt-4o-mini"
temperature = 0.7

with mlflow.start_run(run_name="first-real-llm-experiment"):

    # log params
    mlflow.log_param("model", model_name)
    mlflow.log_param("temperature", temperature)
    mlflow.log_param("prompt_length", len(prompt))
    mlflow.log_param("provider", "AvvalAI")

    # run LLM
    result = llm_call(prompt, model_name, temperature)

    # log metrics
    mlflow.log_metric("tokens_used", result["tokens_used"])
    mlflow.log_metric("latency_ms", result["latency_ms"])

    # log output
    with open("prompt_output.txt", "w", encoding="utf-8") as f:
        f.write(f"PROMPT:\n{prompt}\n\nOUTPUT:\n{result['output']}")

    mlflow.log_artifact("prompt_output.txt")

    print("Run logged successfully!")
    print(f"model:  {model_name}")
    print(f"tokens: {result['tokens_used']}")
    print(f"latency: {result['latency_ms']}ms")

print("\nOpen MLflow UI:")
print("http://localhost:5000")