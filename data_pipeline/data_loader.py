import os
import json
import logging
import re
import hashlib
from typing import Dict, Tuple, Set

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

def is_flattened(repo_id: str, dataset_name: str) -> bool:
    return "generated-queries" in dataset_name or "generated-queries" in repo_id

def clean_text_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_metadata(
    repo_id: str, 
    dataset_name: str, 
    min_length: int = 10, 
    max_length: int = 10000, 
    remove_duplicates: bool = True
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Pass 1: Stream the dataset to extract minimal metadata for sampling.
    Applies Min/Max length filtering and Deduplication via Hashing.
    """
    logger.info(f"Extracting metadata from {repo_id} (Streaming Pass 1)...")
    corpus_meta = {}
    queries_meta = {}
    qrels_dict = {}
    
    seen_hashes = set()
    outlier_count = 0
    duplicate_count = 0

    def process_corpus_row(doc_id: str, text: str):
        nonlocal outlier_count, duplicate_count
        if not doc_id:
            return
            
        cleaned = clean_text_basic(text)
        word_count = len(cleaned.split())
        
        if word_count < min_length or word_count > max_length:
            outlier_count += 1
            return
            
        if remove_duplicates:
            text_hash = hashlib.md5(cleaned.encode('utf-8')).hexdigest()
            if text_hash in seen_hashes:
                duplicate_count += 1
                return
            seen_hashes.add(text_hash)
            
        corpus_meta[doc_id] = word_count

    if is_flattened(repo_id, dataset_name):
        dataset = load_dataset(repo_id, split="train", streaming=True)
        for i, row in enumerate(tqdm(dataset, desc="Streaming flattened metadata")):
            doc_id = str(row.get("_id", row.get("id", f"doc_{i}")))
            text = str(row.get("text", "")).strip()
            query_text = str(row.get("query", "")).strip()

            if doc_id not in corpus_meta:
                process_corpus_row(doc_id, text)

            qid = f"q_{hash(query_text) % 100000000}" if query_text else f"q_{i}"
            if query_text:
                if qid not in queries_meta:
                    queries_meta[qid] = len(clean_text_basic(query_text).split())
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                qrels_dict[qid][doc_id] = 1
    else:
        corpus_ds = load_dataset(repo_id, "corpus", split="corpus", streaming=True)
        for row in tqdm(corpus_ds, desc="Streaming corpus metadata"):
            doc_id = str(row.get("_id", ""))
            text = str(row.get("text", ""))
            process_corpus_row(doc_id, text)

        queries_ds = load_dataset(repo_id, "queries", split="queries", streaming=True)
        for row in tqdm(queries_ds, desc="Streaming queries metadata"):
            qid = str(row.get("_id", ""))
            text = str(row.get("text", row.get("query", "")))
            if qid:
                queries_meta[qid] = len(clean_text_basic(text).split())

        qrels_ds = load_dataset(repo_id, "qrels", split="test", streaming=True)
        for row in tqdm(qrels_ds, desc="Streaming qrels metadata"):
            qid = str(row.get("query-id", row.get("query_id", "")))
            doc_id = str(row.get("corpus-id", row.get("doc_id", "")))
            score = int(row.get("score", 1))
            if qid and doc_id:
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                qrels_dict[qid][doc_id] = score

    logger.info(f"Cleaning Stats: Removed {outlier_count} length outliers, {duplicate_count} duplicates.")
    return corpus_meta, queries_meta, qrels_dict


def save_sampled_data(
    repo_id: str, 
    dataset_name: str, 
    output_dir: str, 
    sampled_qids: Set[str], 
    sampled_doc_ids: Set[str],
    qrels_dict: Dict[str, Dict[str, int]]
):
    """
    Pass 2: Stream the dataset again and save ONLY the sampled records to disk.
    Also applies the basic cleaning to the saved text.
    """
    logger.info(f"Saving sampled data to {output_dir} (Streaming Pass 2)...")
    os.makedirs(output_dir, exist_ok=True)
    
    corpus_path = os.path.join(output_dir, "sampled_corpus.jsonl")
    queries_path = os.path.join(output_dir, "sampled_queries.json")
    qrels_path = os.path.join(output_dir, "sampled_qrels.json")
    
    sampled_queries_dict = {}
    
    # We make a copy of sampled_doc_ids to iterate without modifying the original set
    remaining_docs = set(sampled_doc_ids)
    
    if is_flattened(repo_id, dataset_name):
        dataset = load_dataset(repo_id, split="train", streaming=True)
        with open(corpus_path, "w", encoding="utf-8") as f_corp:
            for i, row in enumerate(tqdm(dataset, desc="Saving sampled flattened data")):
                doc_id = str(row.get("_id", row.get("id", f"doc_{i}")))
                
                if doc_id in remaining_docs:
                    title = str(row.get("title", "")).strip()
                    text = clean_text_basic(str(row.get("text", "")))
                    f_corp.write(json.dumps({"doc_id": doc_id, "title": title, "text": text}) + "\n")
                    remaining_docs.remove(doc_id)

                query_text = str(row.get("query", "")).strip()
                qid = f"q_{hash(query_text) % 100000000}" if query_text else f"q_{i}"
                if query_text and qid in sampled_qids:
                    sampled_queries_dict[qid] = clean_text_basic(query_text)
    else:
        corpus_ds = load_dataset(repo_id, "corpus", split="corpus", streaming=True)
        with open(corpus_path, "w", encoding="utf-8") as f_corp:
            for row in tqdm(corpus_ds, desc="Saving sampled corpus"):
                doc_id = str(row.get("_id", ""))
                if doc_id in sampled_doc_ids:
                    title = str(row.get("title", ""))
                    text = clean_text_basic(str(row.get("text", "")))
                    f_corp.write(json.dumps({"doc_id": doc_id, "title": title, "text": text}) + "\n")

        queries_ds = load_dataset(repo_id, "queries", split="queries", streaming=True)
        for row in tqdm(queries_ds, desc="Saving sampled queries"):
            qid = str(row.get("_id", ""))
            if qid in sampled_qids:
                sampled_queries_dict[qid] = clean_text_basic(str(row.get("text", row.get("query", ""))))

    with open(queries_path, "w", encoding="utf-8") as f_q:
        json.dump(sampled_queries_dict, f_q, ensure_ascii=False, indent=2)

    # Filter qrels using the ORIGINAL sampled_doc_ids set
    sampled_qrels_dict = {
        qid: {did: score for did, score in docs.items() if did in sampled_doc_ids}
        for qid, docs in qrels_dict.items() if qid in sampled_qids
    }
    sampled_qrels_dict = {k: v for k, v in sampled_qrels_dict.items() if v}
    
    with open(qrels_path, "w", encoding="utf-8") as f_qr:
        json.dump(sampled_qrels_dict, f_qr, ensure_ascii=False, indent=2)
        
    logger.info("Sampled dataset saved successfully.")
