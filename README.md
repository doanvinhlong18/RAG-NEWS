# RAG-NEWS

Lightweight RAG pipeline for news retrieval and QA, optimized for low VRAM.

## Multi-Query Retrieval

The pipeline expands a user query into 2-3 variants, retrieves BM25 and dense results independently per variant, and fuses them with Reciprocal Rank Fusion (RRF) before reranking.

## Quick Try

```powershell
python multi_query_retriever_demo.py
```

## Evaluate (Single vs Multi-Query)

```powershell
python evaluation.py --config config.yaml --max_queries 200
```

## Retrieve

```powershell
python retrieval_pipeline.py --query "stock market crash" --config config.yaml --verbose
```

## Dataset Download (HuggingFace)

Set the dataset source in `config.yaml`:

```yaml
dataset:
  source: "huggingface"
  hf_repo: "BeIR/trec-news-generated-queries"
```

Then run preprocessing:

```powershell
python run_pipeline.py
```

For details on the dataset sampling algorithm used to optimize pipeline performance, please refer to our [Sampling Methodology](docs/sampling_methodology.md).
