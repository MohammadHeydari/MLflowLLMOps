import os
import time
import mlflow
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

API_KEY = os.getenv("AVVALAI_API_KEY")
if not API_KEY:
    raise ValueError("AVVALAI_API_KEY is not set in .env")

BASE_URL = "https://api.avalai.ir/v1"

# MLflow setup
mlflow.set_experiment("03-llamaindex-tracking")
mlflow.llama_index.autolog()

# LLM (AvvalAI - OpenAI compatible)
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=API_KEY,
    api_base=BASE_URL,
    temperature=0.3,
)

# Embedding (LOCAL - no API needed)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en"
)

# Documents
documents = [
    Document(text="""
    MLflow is an open-source platform for managing the ML model lifecycle.
    It includes Tracking, Projects, Models, and Model Registry.
    It was built by Databricks in 2018.
    """),
    Document(text="""
    LLMOps refers to practices for deploying and managing LLMs in production.
    It includes monitoring, versioning, evaluation, and continuous improvement.
    Tools include MLflow, LangSmith, and Weights & Biases.
    """),
    Document(text="""
    RAG combines retrieval with generation.
    First it retrieves relevant documents, then passes them to an LLM.
    LlamaIndex is a popular framework for building RAG systems.
    """),
]

# Run experiment
with mlflow.start_run(run_name="llamaindex-rag-pipeline"):

    mlflow.log_param("num_documents", len(documents))
    mlflow.log_param("llm_model", "gpt-4o-mini")
    mlflow.log_param("embedding_model", "BAAI/bge-small-en")
    mlflow.log_param("index_type", "VectorStoreIndex")

    # Build index
    start = time.time()
    index = VectorStoreIndex.from_documents(documents)
    index_time = (time.time() - start) * 1000
    mlflow.log_metric("index_build_time_ms", round(index_time))

    query_engine = index.as_query_engine(similarity_top_k=2)

    questions = [
        "What are the main components of MLflow?",
        "What is the difference between RAG and LLMOps?",
    ]

    for i, question in enumerate(questions):

        start = time.time()
        response = query_engine.query(question)
        latency = (time.time() - start) * 1000

        mlflow.log_metric("query_latency_ms", round(latency), step=i)
        mlflow.log_metric("source_nodes_count", len(response.source_nodes), step=i)

        print("\n" + "=" * 60)
        print(f"Q: {question}")
        print(f"A: {response.response}")
        print(f"Sources used: {len(response.source_nodes)}")
        print(f"Latency: {int(latency)} ms")

print("\nRAG pipeline completed successfully")
print("Open MLflow UI: http://localhost:5000")