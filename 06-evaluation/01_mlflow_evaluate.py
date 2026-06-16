# LLM Evaluation with MLflow
# Built-in metrics: toxicity, ari_grade_level, flesch_kincaid


import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
mlflow.set_experiment("06-llm-evaluation")

# Prepare test data
eval_data = pd.DataFrame({
    "inputs": [
        "What is MLflow?",
        "What does RAG stand for?",
        "When is fine-tuning better than RAG?",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the ML model lifecycle.",
        "RAG stands for Retrieval-Augmented Generation and combines search with an LLM.",
        "When you need domain-specific knowledge or want to change the model's behavior.",
    ],
    "predictions": [
        "MLflow is a tool that helps data scientists track their experiments.",
        "RAG is a method that combines external information with an LLM for better responses.",
        "Fine-tuning is useful when you have enough labeled data and RAG is not sufficient.",
    ]
})

# Run evaluation
with mlflow.start_run(run_name="llm-evaluation-demo"):

    mlflow.log_param("eval_dataset_size", len(eval_data))
    mlflow.log_param("model_under_test", "gpt-4o-mini")

    eval_data.to_csv("eval_dataset.csv", index=False)
    mlflow.log_artifact("eval_dataset.csv")

    results = mlflow.evaluate(
        data=eval_data,
        targets="ground_truth",
        predictions="predictions",
        model_type="text",
        extra_metrics=[
            mlflow.metrics.latency(),
            mlflow.metrics.token_count(),
        ],
    )

    print("Evaluation results:")
    print(results.metrics)

    results.tables["eval_results_table"].to_csv("eval_results.csv", index=False)
    mlflow.log_artifact("eval_results.csv")

print("\nEvaluation complete.")
print("MLflow UI -> 06-llm-evaluation -> check Artifacts")
