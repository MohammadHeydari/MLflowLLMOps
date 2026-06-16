import os
import mlflow
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

mlflow.set_experiment("05-prompt-management")

# AvvalAI client
client = OpenAI(
    api_key=os.getenv("AVVALAI_API_KEY"),
    base_url="https://api.avalai.ir/v1"
)

MODEL_NAME = "gpt-4o-mini"

# Prompt versions
PROMPTS = {
    "summarizer_v1": {
        "version": "1.0.0",
        "description": "Basic summarizer",
        "system": "Summarize the following text.",
        "tags": ["summarization", "basic"],
    },

    "summarizer_v2": {
        "version": "2.0.0",
        "description": "Structured summarizer",
        "system": """Summarize the text in this format:
- Key points (3 bullets)
- Conclusion (1 sentence)""",
        "tags": ["summarization", "structured"],
    },

    "summarizer_v3": {
        "version": "3.0.0",
        "description": "Technical summarizer",
        "system": """You are a technical expert.
Summarize:
1. Core concepts
2. Key insights
3. Practical takeaways""",
        "tags": ["summarization", "technical"],
    },
}

test_text = """
MLflow is an open-source platform built by Databricks.
It is used for tracking ML experiments, managing models, and deployment.
It supports Python, R, and Java.
"""


# LLM call function
def run_llm(system_prompt: str, user_text: str):
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.3
    )

    latency = (time.time() - start) * 1000
    output = response.choices[0].message.content

    return output, latency


print("Testing prompt versions with AvvalAI...\n")

# Evaluate each prompt version
for name, config in PROMPTS.items():

    with mlflow.start_run(run_name=name):

        mlflow.log_param("prompt_name", name)
        mlflow.log_param("version", config["version"])
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("provider", "AvvalAI")

        # tags
        for t in config["tags"]:
            mlflow.set_tag(f"tag_{t}", True)

        # run LLM
        output, latency = run_llm(config["system"], test_text)

        print(f"\n=== {name} ===")
        print(output)

        # logs
        mlflow.log_text(config["system"], "system_prompt.txt")
        mlflow.log_text(test_text, "input.txt")
        mlflow.log_text(output, "output.txt")

        mlflow.log_metric("latency_ms", latency)
        mlflow.log_metric("output_length", len(output))

        print(f"Latency: {latency:.2f} ms")

# Registry snapshot
with mlflow.start_run(run_name="prompt-registry-snapshot"):

    mlflow.log_dict(PROMPTS, "prompt_registry.json")
    mlflow.log_metric("total_prompts", len(PROMPTS))
    mlflow.set_tag("type", "registry_snapshot")
    mlflow.set_tag("snapshot_date", datetime.now().strftime("%Y-%m-%d"))

print("\nPrompt versioning with AvvalAI completed.")
print("Go to MLflow UI -> 05-prompt-management")