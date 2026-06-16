# Run multiple LLM experiments with different parameters 

import os
import mlflow
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("AVVALAI_API_KEY")
if not API_KEY:
    raise ValueError("AVVALAI_API_KEY is not set in .env")

BASE_URL = "https://api.avalai.ir/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

mlflow.set_experiment("01-llm-basics")

#
# Real LLM Call
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


# Experiment Setup
prompt = "Write a Python script to read a CSV file"
model = "gpt-4o-mini"
temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]

print("Running LLM comparison experiments...\n")

# Run Experiments
for temp in temperatures:

    with mlflow.start_run(run_name=f"temp-{temp}"):

        mlflow.log_param("model", model)
        mlflow.log_param("temperature", temp)
        mlflow.log_param("prompt", prompt)
        mlflow.log_param("provider", "AvvalAI")

        result = llm_call(prompt, model, temp)

        mlflow.log_metric("tokens_used", result["tokens_used"])
        mlflow.log_metric("latency_ms", result["latency_ms"])

        # simple proxy for quality (since no ground truth)
        mlflow.log_metric(
            "response_length",
            len(result["output"])
        )

        mlflow.set_tag("experiment_type", "temperature_sweep")

        print(
            f"temperature={temp} | "
            f"tokens={result['tokens_used']} | "
            f"latency={result['latency_ms']}ms"
        )

print("\nOpen MLflow UI:")
print("http://localhost:5000 -> Experiments -> 01-llm-basics")
print("Select runs -> Compare")