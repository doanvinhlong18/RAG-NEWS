import os
import json
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def split_data(
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Splits the sampled queries into Train, Val, Test.
    Ensures qrels follow the query splits to avoid data leakage.
    """
    logger.info(f"Splitting data into Train({train_ratio}) / Val({val_ratio}) / Test({1.0 - train_ratio - val_ratio:.2f})")
    
    queries_path = os.path.join(output_dir, "sampled_queries.json")
    qrels_path = os.path.join(output_dir, "sampled_qrels.json")
    
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
        
    qids = list(queries.keys())
    random.seed(seed)
    random.shuffle(qids)
    
    num_q = len(qids)
    train_end = int(num_q * train_ratio)
    val_end = train_end + int(num_q * val_ratio)
    
    train_qids = qids[:train_end]
    val_qids = qids[train_end:val_end]
    test_qids = qids[val_end:]
    
    splits = {
        "train": train_qids,
        "val": val_qids,
        "test": test_qids
    }
    
    for split_name, split_qids in splits.items():
        split_queries = {qid: queries[qid] for qid in split_qids}
        split_qrels = {qid: qrels[qid] for qid in split_qids if qid in qrels}
        
        with open(os.path.join(output_dir, f"{split_name}_queries.json"), "w", encoding="utf-8") as f:
            json.dump(split_queries, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(output_dir, f"{split_name}_qrels.json"), "w", encoding="utf-8") as f:
            json.dump(split_qrels, f, ensure_ascii=False, indent=2)
            
        logger.info(f"{split_name.capitalize()} Split: {len(split_queries)} queries, {len(split_qrels)} qrels")

    logger.info("Splitting complete. No query leakage across splits.")
