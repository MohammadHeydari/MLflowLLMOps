"""
Lesson 4: Prompt Comparison — the Core of LLMOps
==================================================
What we learn:
  - How to compare multiple prompts against each other
  - Prompt versioning
  - Selecting the best prompt based on metrics

This is one of the most important skills for an LLMOps Engineer.
"""

import mlflow
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
mlflow.set_experiment("02-prompt-comparison")

# ==========================================
# Define multiple prompt versions
# ==========================================
PROMPT_VERSIONS = {
    "v1-basic": {
        "system": "Answer the question.",
        "description": "Simplest possible prompt",
    },
    "v2-role": {
        "system": "You are an MLOps expert with 10 years of experience.",
        "description": "With role definition",
    },
    "v3-structured": {
        "system": """You are an MLOps expert.
Structure your answer as:
1. Short definition
2. Practical example
3. One recommendation""",
        "description": "With explicit structure",
    },
    "v4-cot": {
        "system": """You are an MLOps expert.
Think step by step before answering.
Then give your final answer.""",
        "description": "Chain of Thought",
    },
}

question = "How does the MLflow Model Registry work?"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

print("Starting prompt comparison experiment...\n")
results = []

for version_name, config in PROMPT_VERSIONS.items():
    with mlflow.start_run(run_name=version_name):

        mlflow.log_param("prompt_version", version_name)
        mlflow.log_param("prompt_description", config["description"])
        mlflow.log_param("model", "gpt-4o-mini")
        mlflow.log_param("question", question)

        mlflow.log_text(config["system"], "system_prompt.txt")

        prompt = ChatPromptTemplate.from_messages([
            ("system", config["system"]),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()

        start_time = time.time()
        response = chain.invoke({"question": question})
        latency = (time.time() - start_time) * 1000

        mlflow.log_metric("latency_ms", round(latency))
        mlflow.log_metric("response_length", len(response))
        mlflow.log_metric("word_count", len(response.split()))

        mlflow.log_text(response, "response.txt")

        results.append({
            "version": version_name,
            "latency_ms": round(latency),
            "word_count": len(response.split()),
        })

        print(f"  {version_name}: {round(latency)}ms | {len(response.split())} words")

print("\nSummary:")
print(f"{'Version':<20} {'Latency (ms)':<15} {'Word count'}")
print("-" * 50)
for r in results:
    print(f"{r['version']:<20} {r['latency_ms']:<15} {r['word_count']}")

print("\nFor a visual comparison -> MLflow UI -> select all runs -> Compare Runs")
