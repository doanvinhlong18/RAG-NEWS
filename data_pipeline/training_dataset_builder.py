import os
import json
import logging
import random
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_random_negatives(doc_ids: List[str], positive_ids: List[str], num_neg: int = 5) -> List[str]:
    """Generates random negatives safely."""
    available = list(set(doc_ids) - set(positive_ids))
    if not available:
        return []
    return random.sample(available, min(num_neg, len(available)))


def build_training_datasets(output_dir: str, num_negatives: int = 5):
    """
    Streams split queries and builds Bi-Encoder and Cross-Encoder datasets.
    Fixed: added progress logging + skip if output already exists.
    """
    logger.info("Building Training Datasets (train.jsonl, val.jsonl, test.jsonl)...")

    corpus_path = os.path.join(output_dir, "sampled_corpus.jsonl")

    # --- Count total lines first (fast, no JSON parsing) ---
    logger.info("Counting corpus lines...")
    total_lines = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_lines += 1
    logger.info(f"Corpus has {total_lines:,} documents.")

    # --- Pass 1: Load doc_ids with progress ---
    logger.info("Loading doc_ids from corpus...")
    doc_ids = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            doc_ids.append(json.loads(line)["doc_id"])
            if (i + 1) % 50000 == 0:
                logger.info(f"  Loaded {i+1:,} / {total_lines:,} doc_ids...")
    logger.info(f"Done loading {len(doc_ids):,} doc_ids.")

    splits = ["train", "val", "test"]
    for split in splits:
        out_path = os.path.join(output_dir, f"{split}.jsonl")

        # Skip if already built
        if os.path.exists(out_path):
            logger.info(f"Skipping {split}.jsonl, already exists.")
            continue

        queries_path = os.path.join(output_dir, f"{split}_queries.json")
        qrels_path = os.path.join(output_dir, f"{split}_qrels.json")

        if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
            logger.warning(f"Missing {split} queries or qrels, skipping.")
            continue

        logger.info(f"Building {split} dataset...")

        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels = json.load(f)

        # Build records
        dataset_records = []
        for qid, q_text in queries.items():
            if qid not in qrels:
                continue
            pos_dids = [did for did, score in qrels[qid].items() if score > 0]
            if not pos_dids:
                continue
            neg_dids = generate_random_negatives(doc_ids, pos_dids, num_negatives)
            dataset_records.append({
                "query": q_text,
                "pos_ids": pos_dids,
                "neg_ids": neg_dids,
            })

        logger.info(f"  {split}: {len(dataset_records):,} query records. Fetching texts...")

        # Collect needed doc IDs
        needed_dids = set()
        for r in dataset_records:
            needed_dids.update(r["pos_ids"])
            needed_dids.update(r["neg_ids"])

        # Stream corpus to get texts — with progress
        doc_texts = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc["doc_id"] in needed_dids:
                    doc_texts[doc["doc_id"]] = doc.get("text", "")
                if (i + 1) % 100000 == 0:
                    logger.info(f"  Scanned {i+1:,} / {total_lines:,} docs, found {len(doc_texts):,} texts so far...")

        logger.info(f"  Fetched {len(doc_texts):,} doc texts.")

        # Write output
        written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in dataset_records:
                pos_texts = [doc_texts.get(did, "") for did in r["pos_ids"] if did in doc_texts]
                neg_texts = [doc_texts.get(did, "") for did in r["neg_ids"] if did in doc_texts]
                if pos_texts:
                    f.write(json.dumps({
                        "query": r["query"],
                        "positives": pos_texts,
                        "negatives": neg_texts,
                    }) + "\n")
                    written += 1

        logger.info(f"Saved {written:,} records to {out_path}")

    logger.info("All training datasets built successfully.")