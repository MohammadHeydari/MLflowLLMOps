# LLMOps with MLflow

> A hands-on learning repository for LLMOps using MLflow for tracking, evaluation, and deployment of language models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-orange.svg)](https://mlflow.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)

---

## Goal

This repository covers the practical learning path for becoming an **LLMOps Engineer**:

- Tracking LLM experiments with MLflow
- Comparing prompts, models, and parameters
- Automated evaluation of LLM outputs
- Managing the model lifecycle with Model Registry
- Preparing models for production

---


## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/MohammadHeydari/llmops-mlflow.git
cd llmops-mlflow
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Open .env and fill in your keys
```

### 3. Start the MLflow UI

```bash
mlflow ui
# Then go to: http://localhost:5000
```

### 4. Run your first experiment

```bash
python 01-mlflow-basics/01_first_experiment.py
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Experiment** | A logical container for related runs (e.g. "Which prompt performs best?") |
| **Run** | A single execution with specific parameters |
| **Parameters** | Inputs to the experiment (model_name, temperature, prompt) |
| **Metrics** | Measurable outputs (latency, cost, accuracy) |
| **Artifacts** | Saved files (models, prompts, outputs) |
| **Model Registry** | Version control system for models |

---

## Stack

- **MLflow** — Experiment tracking, Model Registry, Evaluation
- **LangChain** — LLM chains, RAG, Agents
- **LlamaIndex** — Document indexing, Query engines
- **HuggingFace** — Open-source models
- **OpenAI / Groq** — LLM providers

---

## Learning Path

```
01-basics → 02-langchain → 03-llamaindex → 04-huggingface
                                                    |
               07-production <- 06-evaluation <- 05-prompts
```

---

*Built for learning LLMOps Engineering*
