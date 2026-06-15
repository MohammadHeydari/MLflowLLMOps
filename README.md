# 🚀 LLMOps with MLflow

> یادگیری عملی LLMOps با استفاده از MLflow برای tracking، evaluation و deployment مدل‌های زبانی

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-orange.svg)](https://mlflow.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)

---

## 🎯 هدف این ریپازیتوری

این ریپازیتوری مسیر یادگیری **LLMOps Engineer** رو به‌صورت عملی پوشش می‌ده:
- ثبت و tracking آزمایش‌های LLM با MLflow
- مقایسه promptها، مدل‌ها و پارامترها
- evaluation خودکار خروجی‌های LLM
- مدیریت چرخه حیات مدل (Model Registry)
- آماده‌سازی برای production

---

## 🗺 ساختار ریپازیتوری

```
llmops-mlflow/
├── 01-mlflow-basics/          # پایه MLflow — بدون LLM
│   ├── 01_first_experiment.py
│   ├── 02_compare_runs.py
│   └── README.md
│
├── 02-langchain-tracking/     # LangChain + MLflow
│   ├── 01_simple_chain.py
│   ├── 02_rag_pipeline.py
│   ├── 03_compare_prompts.py
│   └── README.md
│
├── 03-llamaindex-tracking/    # LlamaIndex + MLflow
│   ├── 01_index_and_query.py
│   ├── 02_track_retrieval.py
│   └── README.md
│
├── 04-huggingface-tracking/   # HuggingFace + MLflow
│   ├── 01_log_hf_model.py
│   ├── 02_fine_tune_track.py
│   └── README.md
│
├── 05-prompt-management/      # Prompt Versioning
│   ├── prompts/
│   ├── 01_version_prompts.py
│   └── README.md
│
├── 06-evaluation/             # LLM Evaluation با MLflow
│   ├── 01_mlflow_evaluate.py
│   ├── 02_custom_metrics.py
│   └── README.md
│
├── 07-production/             # Serve و Deploy
│   ├── 01_register_model.py
│   ├── 02_serve_llm.py
│   └── README.md
│
├── requirements.txt
└── README.md                  # همین فایل
```

---

## ⚡ شروع سریع

### ۱. نصب dependencies

```bash
git clone https://github.com/YOUR_USERNAME/llmops-mlflow.git
cd llmops-mlflow
pip install -r requirements.txt
```

### ۲. تنظیم API Key

```bash
cp .env.example .env
# فایل .env رو باز کن و کلیدهات رو بنویس
```

### ۳. اجرای MLflow UI

```bash
mlflow ui
# بعد برو به: http://localhost:5000
```

### ۴. اولین آزمایش

```bash
python 01-mlflow-basics/01_first_experiment.py
```

---

## 🔑 مفاهیم کلیدی

| مفهوم | توضیح |
|-------|--------|
| **Experiment** | یه پروژه یا سوال مشخص (مثلاً: "کدوم prompt بهتره؟") |
| **Run** | یه بار اجرای آزمایش با پارامترهای مشخص |
| **Parameters** | ورودی‌های آزمایش (model_name, temperature, prompt) |
| **Metrics** | نتایج قابل اندازه‌گیری (latency, cost, accuracy) |
| **Artifacts** | فایل‌های ذخیره‌شده (مدل، prompt، output) |
| **Model Registry** | سیستم مدیریت نسخه‌های مدل |

---

## 🛠 Stack

- **MLflow** — Experiment tracking, Model Registry, Evaluation
- **LangChain** — LLM chains, RAG, Agents
- **LlamaIndex** — Document indexing, Query engines
- **HuggingFace** — Open-source models
- **OpenAI / Groq** — LLM providers

---

## 📈 مسیر یادگیری

```
01-basics → 02-langchain → 03-llamaindex → 04-huggingface
                                                    ↓
               07-production ← 06-evaluation ← 05-prompts
```

---

*ساخته‌شده برای یادگیری LLMOps Engineering*
