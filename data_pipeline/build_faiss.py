"""
build_faiss.py
==============
Standalone script to (re)build FAISS index from corpus_chunks.jsonl.

Usage:
    # Build with base bi-encoder (first time)
    python build_faiss.py

    # Rebuild with finetuned bi-encoder (after bi_encoder_training.py)
    python build_faiss.py --model models/bi_encoder

    # Explicit config
    python build_faiss.py --config config.yaml --model models/bi_encoder

The script always backs up the existing index before overwriting.
"""

import os
import json
import pickle
import shutil
import logging
import argparse
from typing import Optional

import numpy as np
import faiss
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Chunk generator
# ---------------------------------------------------------------------------

def chunk_generator(filepath: str, batch_size: int):
    """Yield batches of chunk dicts from JSONL."""
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


# ---------------------------------------------------------------------------
# Backup helper
# ---------------------------------------------------------------------------

def backup_existing_index(index_dir: str) -> None:
    """
    Rename existing faiss.index → faiss_prev.index before overwriting.
    chunk_metadata.pkl is NOT backed up — it stays the same.
    """
    index_path = os.path.join(index_dir, "faiss.index")
    backup_path = os.path.join(index_dir, "faiss_prev.index")

    if os.path.exists(index_path):
        shutil.copy2(index_path, backup_path)
        logger.info("Backed up existing index → %s", backup_path)
    else:
        logger.info("No existing FAISS index to back up.")


# ---------------------------------------------------------------------------
# Build FAISS
# ---------------------------------------------------------------------------

def build_faiss_index(
    output_dir: str,
    model_name: str,
    batch_size: int = 256,
) -> None:
    """
    Stream corpus_chunks.jsonl, encode with model_name, build IndexFlatIP.

    Saves:
        faiss.index         — vector index (overwritten)
        chunk_metadata.pkl  — {faiss_idx: {chunk_id, doc_id}} (overwritten)

    chunk_metadata is always rebuilt to guarantee consistency with the
    current corpus_chunks.jsonl, even if it hasn't changed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading encoder: %s on %s", model_name, device)

    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()   # fp16 to save VRAM

    chunks_path = os.path.join(output_dir, "corpus_chunks.jsonl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"corpus_chunks.jsonl not found: {chunks_path}")

    # Count total chunks for progress bar
    logger.info("Counting chunks …")
    total_chunks = sum(1 for line in open(chunks_path, encoding="utf-8") if line.strip())
    logger.info("Total chunks: %d", total_chunks)

    # Get embedding dimension
    dim = model.encode(["test"], convert_to_numpy=True).shape[1]
    logger.info("Embedding dim: %d", dim)

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
        logger.info("Using HNSW index with M=%d", hnsw_m)
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    else:
        logger.info("Using Flat index")
        index = faiss.IndexFlatIP(dim)
    chunk_metadata = {}
    total_added = 0

    pbar = tqdm(total=total_chunks, desc="Encoding & indexing", unit="chunks")

    for batch in chunk_generator(chunks_path, batch_size):
        texts = [c["chunk_text"] for c in batch]

        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM — retrying batch on CPU")
                torch.cuda.empty_cache()
                model_cpu = model.to("cpu").float()
                embeddings = model_cpu.encode(
                    texts,
                    batch_size=max(1, batch_size // 4),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                model = model.to(device)
                if device == "cuda":
                    model = model.half()
            else:
                raise

        embeddings = embeddings.astype(np.float32)
        index.add(embeddings)

        for i, c in enumerate(batch):
            chunk_metadata[total_added + i] = {
                "chunk_id": c["chunk_id"],
                "doc_id":   c["doc_id"],
            }

        total_added += len(batch)
        pbar.update(len(batch))

    pbar.close()

    # Save
    faiss_path = os.path.join(output_dir, "faiss.index")
    meta_path  = os.path.join(output_dir, "chunk_metadata.pkl")

    faiss.write_index(index, faiss_path)
    logger.info("FAISS index saved → %s (%d vectors)", faiss_path, total_added)

    with open(meta_path, "wb") as f:
        pickle.dump(chunk_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("chunk_metadata saved → %s", meta_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build or rebuild FAISS index.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name or path to use for encoding. "
            "Defaults to bi_encoder_training.output_path if it exists, "
            "otherwise falls back to models.bi_encoder_base."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encoding batch size. Defaults to indexing.batch_size in config.",
    )
    args = parser.parse_args()

    cfg       = load_config(args.config)
    index_dir = cfg["paths"]["index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    # Resolve model
    if args.model:
        model_name = args.model
    else:
        finetuned_path = cfg["bi_encoder_training"]["output_path"]
        if os.path.exists(os.path.join(finetuned_path, "config.json")):
            model_name = finetuned_path
            logger.info("Auto-detected finetuned bi-encoder: %s", model_name)
        else:
            model_name = cfg["models"]["bi_encoder_base"]
            logger.info("Finetuned bi-encoder not found — using base: %s", model_name)

    batch_size = args.batch_size or cfg["indexing"]["batch_size"]

    logger.info("=" * 60)
    logger.info("FAISS rebuild")
    logger.info("  index_dir  : %s", index_dir)
    logger.info("  model      : %s", model_name)
    logger.info("  batch_size : %d", batch_size)
    logger.info("=" * 60)

    # Backup existing index
    backup_existing_index(index_dir)

    # Build
    build_faiss_index(
        output_dir=index_dir,
        model_name=model_name,
        batch_size=batch_size,
    )

    logger.info("\nFAISS index rebuilt successfully.")
    logger.info(
        "Next step: delete CE cache and retrain CE\n"
        "  del outputs/pipeline/ce_hard_negatives_v4.jsonl\n"
        "  python cross_encoder_training.py"
    )


if __name__ == "__main__":
    main()