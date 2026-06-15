"""
Lesson 2: Comparing Multiple Runs
===================================
What we learn:
  - How to run multiple experiments with different parameters
  - How to find the best result
  - This pattern is the foundation of LLMOps:
    "Which model / prompt / temperature works best?"

Run:
  python 01-mlflow-basics/02_compare_runs.py
"""

import mlflow
import random

mlflow.set_experiment("01-llm-basics")

# ==========================================
# Core question: which temperature is best?
# This is something LLMOps engineers do every day.
# ==========================================

def fake_llm_call(prompt: str, model: str, temperature: float) -> dict:
    """Simulate an LLM with realistic behavior"""
    random.seed(int(temperature * 10))  # for reproducibility
    base_quality = 0.85 - abs(temperature - 0.7) * 0.3
    return {
        "output": f"Response with temperature={temperature}",
        "tokens_used": random.randint(80, 120),
        "latency_ms": random.randint(800, 1200),
        "quality_score": round(base_quality + random.uniform(-0.05, 0.05), 3),
    }


prompt = "Write a Python script to read a CSV file"
model = "gpt-4o-mini"
temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]

print("Running comparison experiments...\n")

for temp in temperatures:
    with mlflow.start_run(run_name=f"temp-{temp}"):

        mlflow.log_param("model", model)
        mlflow.log_param("temperature", temp)
        mlflow.log_param("prompt", prompt)

        result = fake_llm_call(prompt, model, temp)

        mlflow.log_metric("tokens_used", result["tokens_used"])
        mlflow.log_metric("latency_ms", result["latency_ms"])
        mlflow.log_metric("quality_score", result["quality_score"])

        mlflow.set_tag("experiment_type", "temperature_sweep")

        print(f"  temperature={temp} | quality={result['quality_score']} | latency={result['latency_ms']}ms")

print("\nOpen MLflow UI to see the comparison chart:")
print("  http://localhost:5000 -> Experiments -> 01-llm-basics")
print("  Select all runs -> Compare")
