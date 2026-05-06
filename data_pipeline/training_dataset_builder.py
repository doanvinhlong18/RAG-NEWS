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
    Uses random negatives for speed/memory efficiency since BM25 isn't queried per row.
    """
    logger.info("Building Training Datasets (train.jsonl, val.jsonl, test.jsonl)...")
    
    # Load all doc IDs for random negative sampling
    # We only need IDs, not the full text
    doc_ids = []
    corpus_path = os.path.join(output_dir, "sampled_corpus.jsonl")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc_ids.append(json.loads(line)["doc_id"])
                
    # Build dictionary of texts for cross encoder pairs
    # Wait, Cross Encoder needs the actual text!
    # If we can't load all text to memory, we'll need an efficient lookup.
    # We'll use random access file pointer or load it if it fits (it's 50%).
    # To strictly follow anti-OOM, we do a 2-pass:
    # Pass 1: generate pairs (qid, pos_did, neg_did)
    # Pass 2: stream corpus and populate texts.
    
    splits = ["train", "val", "test"]
    for split in splits:
        queries_path = os.path.join(output_dir, f"{split}_queries.json")
        qrels_path = os.path.join(output_dir, f"{split}_qrels.json")
        
        if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
            continue
            
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels = json.load(f)
            
        # Build structure
        dataset_records = []
        for qid, q_text in queries.items():
            if qid not in qrels:
                continue
                
            pos_dids = [did for did, score in qrels[qid].items() if score > 0]
            if not pos_dids:
                continue
                
            neg_dids = generate_random_negatives(doc_ids, pos_dids, num_negatives)
            
            # Bi-Encoder format typically wants list of [query, pos, neg1, neg2...]
            # Or dict: {"query": q, "pos": [d1], "neg": [n1, n2]}
            dataset_records.append({
                "query": q_text,
                "pos_ids": pos_dids,
                "neg_ids": neg_dids
            })
            
        # Now stream corpus to fill in texts for pos_ids and neg_ids
        needed_dids = set()
        for r in dataset_records:
            needed_dids.update(r["pos_ids"])
            needed_dids.update(r["neg_ids"])
            
        doc_texts = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc["doc_id"] in needed_dids:
                    doc_texts[doc["doc_id"]] = doc.get("text", "")
                    
        # Write to jsonl
        out_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in dataset_records:
                pos_texts = [doc_texts.get(did, "") for did in r["pos_ids"] if did in doc_texts]
                neg_texts = [doc_texts.get(did, "") for did in r["neg_ids"] if did in doc_texts]
                
                if pos_texts:
                    f.write(json.dumps({
                        "query": r["query"],
                        "positives": pos_texts,
                        "negatives": neg_texts
                    }) + "\n")
                    
        logger.info(f"Saved {len(dataset_records)} records to {out_path}")
