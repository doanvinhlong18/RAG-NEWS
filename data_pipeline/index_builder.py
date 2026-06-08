import os
import json
import pickle
import logging
from typing import List

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def chunk_generator(filepath: str, batch_size: int):
    """Yields batches of chunks from JSONL."""
    batch = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def build_bm25_index(output_dir: str):
    """
    Builds BM25 index. To avoid full text memory explosion, 
    we use a generator to stream tokenized arrays.
    """
    logger.info("Building BM25 index from tokenized chunks via streaming...")
    from data_pipeline.sparse_bm25 import SparseBM25
    
    chunks_path = os.path.join(output_dir, "corpus_chunks.jsonl")
    
    def token_generator():
        count = 0
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                yield chunk.get("chunk_tokens", [])
                count += 1
                if count % 100000 == 0:
                    logger.info(f"Streamed {count} chunks for BM25...")
                    
    # Use our memory-efficient SparseBM25
    bm25 = SparseBM25()
    bm25.fit(token_generator())
    
    bm25_path = os.path.join(output_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
        
    logger.info(f"BM25 index saved to {bm25_path} (corpus size: {bm25.corpus_size})")
    return bm25_path

def build_faiss_index(output_dir: str, model_name: str, batch_size: int = 256):
    """
    Streams chunks, encodes them in batches, and adds to IndexFlatIP.
    Prevents holding massive (N, 384) arrays in RAM.
    """
    logger.info("Building FAISS IndexFlatIP via batched streaming...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading encoder {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()
        
    chunks_path = os.path.join(output_dir, "corpus_chunks.jsonl")
    
    # Initialize index - we need the dimension first
    dummy_emb = model.encode(["test"])
    dim = dummy_emb.shape[1]
    
    # Load config to get indexing parameters
    import yaml
    config_path = "config.yaml"
    cfg = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Error loading config.yaml: {e}")
            
    idx_cfg = cfg.get("indexing", {})
    index_type = idx_cfg.get("faiss_index_type", "flat")
    hnsw_m = idx_cfg.get("hnsw_m", 32)

    if index_type == "hnsw":
        logger.info(f"Using HNSW index with M={hnsw_m}")
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    else:
        logger.info("Using Flat index")
        index = faiss.IndexFlatIP(dim)
    
    total_added = 0
    chunk_metadata = {}
    
    for batch in chunk_generator(chunks_path, batch_size):
        texts = [c["chunk_text"] for c in batch]
        
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM, retrying batch on CPU")
                torch.cuda.empty_cache()
                model_cpu = model.to("cpu").float()
                embeddings = model_cpu.encode(
                    texts,
                    batch_size=max(1, batch_size // 4),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                model = model.to("cuda").half() # back to gpu for next batch
            else:
                raise
                
        embeddings = embeddings.astype(np.float32)
        index.add(embeddings)
        
        # Save metadata (ID mapping)
        for i, c in enumerate(batch):
            chunk_metadata[total_added + i] = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"]
            }
            
        total_added += len(batch)
        if total_added % 10000 == 0:
            logger.info(f"Added {total_added} vectors to FAISS index.")
            
    faiss_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, faiss_path)
    
    meta_path = os.path.join(output_dir, "chunk_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(chunk_metadata, f)
        
    logger.info(f"FAISS index saved to {faiss_path} with {total_added} vectors.")
