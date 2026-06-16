import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVVALAI_API_KEY")
if not API_KEY:
    raise ValueError("AVVALAI_API_KEY is not set in .env")

BASE_URL = "https://api.avalai.ir/v1"

mlflow.langchain.autolog()
mlflow.set_experiment("02-langchain-tracking")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Give short, precise answers."),
    ("human", "{question}"),
])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
)

chain = prompt_template | llm | StrOutputParser()

with mlflow.start_run(run_name="langchain-simple-chain"):

    domain = "artificial intelligence"
    question = "What is the difference between RAG and Fine-tuning?"

    mlflow.log_param("domain", domain)
    mlflow.log_param("question_type", "conceptual")
    mlflow.log_param("provider", "AvvalAI")

    print(f"\nQuestion: {question}\n")
    print("Fetching response...\n")

    response = chain.invoke({
        "domain": domain,
        "question": question,
    })

    print(f"Response:\n{response}\n")

    mlflow.log_text(response, "response.txt")
    mlflow.log_metric("response_length", len(response))

print("Chain executed and logged to MLflow")
print("Open UI: http://localhost:5000 -> 02-langchain-tracking")