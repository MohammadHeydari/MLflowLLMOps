import mlflow
import time

mlflow.set_experiment("01-llm-basics")


def fake_llm_call(prompt: str, model: str, temperature: float) -> dict:
    
    time.sleep(0.1)  

    return {
        "output": f"[{model}] Response to: {prompt[:30]}...",
        "tokens_used": len(prompt.split()) * 2,
        "latency_ms": 100,
    }


prompt = "تفاوت Machine Learning و Deep Learning رو توضیح بده"
model_name = "gpt-4o-mini"
temperature = 0.7

with mlflow.start_run(run_name="First Experiments: "):

    mlflow.log_param("model", model_name)
    mlflow.log_param("temperature", temperature)
    mlflow.log_param("prompt_length", len(prompt))


    result = fake_llm_call(prompt, model_name, temperature)


    mlflow.log_metric("tokens_used", result["tokens_used"])
    mlflow.log_metric("latency_ms", result["latency_ms"])


    with open("prompt_output.txt", "w", encoding="utf-8") as f:
        f.write(f"PROMPT:\n{prompt}\n\nOUTPUT:\n{result['output']}")
    mlflow.log_artifact("prompt_output.txt")

    print("Run ")
    print(f"   Model: {model_name}")
    print(f"   Token: {result['tokens_used']}")
    print(f"   Time: {result['latency_ms']}ms")

print("\n mlflow ui")
print("http://localhost:5000")
