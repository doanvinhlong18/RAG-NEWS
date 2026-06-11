"""
compare_models.py
So sánh LLaMA vs Qwen trên cùng một bộ queries.
Kết quả lưu vào results/model_comparison.json

Cách dùng:
    python compare_models.py --model llama
    python compare_models.py --model qwen
"""
from dotenv import load_dotenv
load_dotenv()

import json
import time
import os
import gc
import argparse
import torch

QUERIES = [
    "Who won the 2016 US election?",
    "Who is Elon Musk?",
    "What caused the 2008 financial crisis?",
    "What is SpaceX?",
    "Who is Hillary Clinton?",
]

MODELS = {
    "llama": "llama-3.3-70b-versatile",
    "qwen":  "qwen/qwen3-32b",
}

CONFIG_PATH = "config.yaml"
OUTPUT_PATH = "results/model_comparison.json"


def set_model(config_path: str, model_name: str):
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("primary_model:"):
            indent = len(line) - len(line.lstrip())
            lines[i] = " " * indent + f'primary_model: "{model_name}"\n'
            break
    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f">> Set primary_model = {model_name}")


def run_query(pipeline, query: str) -> dict:
    t0 = time.time()
    result = pipeline.run(query)
    total = time.time() - t0

    return {
        "query":               query,
        "answer":              result.get("answer", ""),
        "latency_retrieval_s": round(getattr(pipeline.retriever, "last_hybrid_time", 0), 3),
        "latency_llm_s":       round(result.get("llm_time", 0), 3),
        "latency_total_s":     round(total, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "qwen"], required=True)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    model_name = MODELS[args.model]
    set_model(CONFIG_PATH, model_name)

    # Reload config sau khi đổi
    import importlib
    import rag_inference
    importlib.reload(rag_inference)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    pipeline = rag_inference.RAGPipeline(CONFIG_PATH)

    records = []
    for q in QUERIES:
        print(f"\nQuery: {q}")
        r = run_query(pipeline, q)
        print(f"  Answer   : {r['answer'][:150]}...")
        print(f"  Retrieval: {r['latency_retrieval_s']}s")
        print(f"  LLM      : {r['latency_llm_s']}s")
        print(f"  Total    : {r['latency_total_s']}s")
        records.append(r)

    result_data = {
        args.model: {
            "model":            model_name,
            "queries":          records,
            "avg_retrieval_s":  round(sum(r["latency_retrieval_s"] for r in records) / len(records), 3),
            "avg_llm_s":        round(sum(r["latency_llm_s"] for r in records) / len(records), 3),
            "avg_total_s":      round(sum(r["latency_total_s"] for r in records) / len(records), 3),
        }
    }

    # Load existing results nếu có rồi merge
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(result_data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    # In summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Avg Retrieval':>15} {'Avg LLM':>10} {'Avg Total':>12}")
    print("-" * 52)
    for key, data in existing.items():
        print(f"{key:<12} {data['avg_retrieval_s']:>14}s {data['avg_llm_s']:>9}s {data['avg_total_s']:>11}s")

    print(f"\nĐã lưu → {OUTPUT_PATH}")

    # Cleanup
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()