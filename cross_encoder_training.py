"""
cross_encoder_training.py
==========================
Fine-tune Cross-Encoder (MiniLM-L6-v2) on BEIR trec-news with hard negatives.

NOTE: Disabled by default (cross_encoder_training.enabled=false in config.yaml).
      Pretrained ms-marco-MiniLM-L-6-v2 already achieves strong performance on
      news domain. Enable only if you have domain-specific labeled pairs.

IMPORTANT: Uses CrossEncoder.fit() standard API — NO manual AMP/GradScaler
           (avoids the NaN gradient / ValueError bug from manual torch.cuda.amp).

Memory:
  - batch_size=4, grad_accum=4 → effective=16
  - fp16 via native use_amp
  - max_seq_length=256
  - Peak VRAM ~2.0GB on RTX 3050 Ti

Run:
  python cross_encoder_training.py
  python cross_encoder_training.py --config config.yaml
"""

import os
import random
import logging
import argparse
import pickle
import json
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_ce_cache(cache_path: str) -> List[Tuple[str, str, float]]:
    if not os.path.exists(cache_path):
        return []
    samples: List[Tuple[str, str, float]] = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            samples.append((item["query"], item["doc"], float(item["label"])))
    return samples


def save_ce_cache(cache_path: str, examples: List) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {"query": ex.texts[0], "doc": ex.texts[1], "label": ex.label},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training Data Construction
# ---------------------------------------------------------------------------
def build_ce_training_data(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    chunks: List[Dict],
    top_k_neg: int = 20,
    max_samples: int = 100000,
    index_dir: str = "outputs/pipeline",
    bi_encoder_model_name: str = "msmarco-distilbert-base-tas-b",
) -> List:
    """
    Build Cross-Encoder training pairs using FAISS hard negatives.

    Format: InputExample(texts=[query, doc_text], label=0.0 or 1.0)

    Strategy:
      - For each query: 1 positive (from qrels) + N negatives (FAISS top-K, non-relevant)
      - Balance: ~50% positive, ~50% negative
    """
    from sentence_transformers import InputExample, SentenceTransformer
    import numpy as np

    logger.info("Loading FAISS index for CE hard negative mining …")
    try:
        index, faiss_doc_ids = load_faiss_resources(index_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return []

    # Use bi-encoder to embed queries for ANN retrieval
    bi_encoder_model = SentenceTransformer(bi_encoder_model_name)

    # Normalize chunks (load_chunks may return dict or list of dicts)
    normalized_chunks: List[Dict] = []
    chunk_id_to_text: Dict[str, str] = {}
    if isinstance(chunks, dict):
        for chunk_id, chunk_text in tqdm(chunks.items(), desc="Loading chunk texts"):
            if chunk_id and chunk_text:
                chunk_id_to_text[str(chunk_id)] = chunk_text
    else:
        for c in tqdm(chunks, desc="Normalizing chunks"):
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except json.JSONDecodeError:
                    c = None
            if not isinstance(c, dict):
                c = None
            normalized_chunks.append(c)

        for c in normalized_chunks:
            if not c:
                continue
            chunk_id = c.get("chunk_id")
            chunk_text = c.get("chunk_text")
            if chunk_id and chunk_text:
                chunk_id_to_text[str(chunk_id)] = chunk_text

    # doc_id → first chunk text (CE works on chunk text)
    doc_id_to_chunk_text: Dict[str, str] = {}
    if faiss_doc_ids:
        for doc_id, chunk_id in faiss_doc_ids:
            if doc_id is None:
                continue
            doc_id_key = str(doc_id)
            if doc_id_key in doc_id_to_chunk_text:
                continue
            if chunk_id is not None:
                chunk_text = chunk_id_to_text.get(str(chunk_id))
                if chunk_text:
                    doc_id_to_chunk_text[doc_id_key] = chunk_text

    # Fallback for list-of-dict chunks if FAISS mapping is sparse
    if not doc_id_to_chunk_text and normalized_chunks:
        for c in normalized_chunks:
            if not c:
                continue
            doc_id = c.get("doc_id")
            chunk_text = c.get("chunk_text")
            if doc_id and chunk_text:
                doc_id_to_chunk_text.setdefault(str(doc_id), chunk_text)

    logger.info(
        "Mapping stats: chunk_texts=%d | faiss_map=%d | doc_id_texts=%d",
        len(chunk_id_to_text),
        len(faiss_doc_ids),
        len(doc_id_to_chunk_text),
    )

    examples = []
    qids = [q for q in qrels if q in queries]
    random.shuffle(qids)

    # Pre-encode queries for FAISS search
    query_texts = [queries[qid] for qid in qids]
    query_emb = bi_encoder_model.encode(
        query_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_emb = np.asarray(query_emb, dtype="float32")

    logger.info("Searching FAISS for hard negatives …")
    top_k = min(max(int(top_k_neg), 10), 50)
    scores, indices = index.search(query_emb, top_k)

    pos_found = 0
    for row_idx, qid in enumerate(tqdm(qids, desc="Building CE pairs")):
        if len(examples) >= max_samples:
            break

        query_text = queries[qid]
        pos_doc_ids = {str(did) for did in qrels[qid].keys()}

        # Positive example
        pos_text = None
        for doc_id in pos_doc_ids:
            pos_text = doc_id_to_chunk_text.get(doc_id)
            if pos_text:
                break
        if not pos_text:
            continue
        pos_found += 1
        examples.append(InputExample(texts=[query_text, pos_text], label=1.0))

        # Hard negatives via FAISS ANN
        neg_count = 0
        for idx in indices[row_idx]:
            if idx < 0 or idx >= len(faiss_doc_ids):
                continue
            doc_id, chunk_id = faiss_doc_ids[idx]
            doc_id_key = str(doc_id) if doc_id is not None else None
            if not doc_id_key:
                continue
            if doc_id_key in pos_doc_ids:
                continue
            neg_text = None
            if chunk_id is not None:
                neg_text = chunk_id_to_text.get(str(chunk_id))
            if not neg_text:
                neg_text = doc_id_to_chunk_text.get(doc_id_key)
            if not neg_text:
                continue
            examples.append(InputExample(texts=[query_text, neg_text], label=0.0))
            neg_count += 1
            if neg_count >= 2:  # 2 hard negatives per positive
                break

    logger.info(
        "Positive matches: %d/%d queries", pos_found, len(qids)
    )
    random.shuffle(examples)
    logger.info(f"Built {len(examples):,} CE training pairs "
                f"({sum(1 for e in examples if e.label == 1.0)} pos, "
                f"{sum(1 for e in examples if e.label == 0.0)} neg)")
    return examples


def load_faiss_resources(index_dir: str) -> Tuple["faiss.Index", List[Tuple[str, str]]]:
    """Load FAISS index and mapping used for ANN retrieval."""
    import faiss

    index_path = os.path.join(index_dir, "faiss.index")
    doc_ids_path = os.path.join(index_dir, "doc_ids.pkl")
    meta_path = os.path.join(index_dir, "chunk_metadata.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)

    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            chunk_metadata = pickle.load(f)

        # chunk_metadata: idx -> {chunk_id, doc_id}
        max_idx = max(chunk_metadata.keys()) if chunk_metadata else -1
        mapping: List[Tuple[str, str]] = [(None, None)] * (max_idx + 1)
        for idx, meta in chunk_metadata.items():
            mapping[idx] = (meta.get("doc_id"), meta.get("chunk_id"))

        return index, mapping

    if os.path.exists(doc_ids_path):
        with open(doc_ids_path, "rb") as f:
            doc_ids = pickle.load(f)
        # Normalize to (doc_id, chunk_id) tuples
        normalized = [(doc_id, None) for doc_id in doc_ids]
        return index, normalized

    raise FileNotFoundError(
        f"No doc_id mapping found. Expected {meta_path} or {doc_ids_path}."
    )


# ---------------------------------------------------------------------------
# CE Evaluator
# ---------------------------------------------------------------------------
def build_ce_evaluator(examples: List, eval_fraction: float = 0.2):
    """Split off a held-out evaluation set for CE training."""
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

    n_eval = max(10, int(len(examples) * eval_fraction))
    eval_examples = examples[-n_eval:]

    sentences1 = [e.texts[0] for e in eval_examples]
    sentences2 = [e.texts[1] for e in eval_examples]
    labels     = [int(e.label) for e in eval_examples]

    return CEBinaryClassificationEvaluator(
        sentence_pairs=list(zip(sentences1, sentences2)),
        labels=labels,
        name="ce-eval",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_cross_encoder(examples: List, evaluator, cfg: dict, output_path: str):
    """
    Train Cross-Encoder using standard CrossEncoder.fit() API.

    DELIBERATELY uses standard API (not manual GradScaler) to avoid:
      - ValueError from mismatched AMP states
      - NaN gradients from improper mixed-precision handling
    """
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    ce_cfg = cfg["cross_encoder_training"]

    # ── Model ────────────────────────────────────────────────────────────────
    # Resume from fine-tuned if available, else use base
    is_resuming = os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "config.json"))

    if is_resuming:
        logger.info(f"Resuming from checkpoint: {output_path}")
        model = CrossEncoder(output_path, max_length=ce_cfg["max_seq_length"])
    else:
        logger.info(f"Initializing CE model: {ce_cfg['base_model']}")
        model = CrossEncoder(
            ce_cfg["base_model"],
            num_labels=1,
            max_length=ce_cfg["max_seq_length"],
        )

    # Train/eval split (don't include eval examples in training)
    n_eval = max(10, int(len(examples) * 0.1))
    train_examples = examples[:-n_eval]

    total_steps = (len(train_examples) // ce_cfg["batch_size"]) * ce_cfg["num_epochs"]
    # Khi resume: bỏ warmup, giảm LR xuống 30%
    if is_resuming:
        warmup_steps = 0
        effective_lr = ce_cfg["learning_rate"] * 0.3  # 1e-5 * 0.3 = 3e-6
        logger.info(f"Resume mode: warmup=0, lr={effective_lr:.2e}")
    else:
        warmup_steps = int(total_steps * ce_cfg["warmup_ratio"])
        effective_lr = ce_cfg["learning_rate"]

    logger.info(
        f"CE Training: {len(train_examples):,} pairs | "
        f"batch={ce_cfg['batch_size']} | "
        f"epochs={ce_cfg['num_epochs']} | "
        f"warmup={warmup_steps} steps | "
        f"fp16={ce_cfg['use_amp']}"
    )

    model.fit(
        train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=ce_cfg["batch_size"]),
        evaluator=evaluator,
        epochs=ce_cfg["num_epochs"],
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=True,
        optimizer_params={"lr": ce_cfg["learning_rate"]},
        use_amp=ce_cfg["use_amp"],          # Standard API — no manual GradScaler
        # gradient_accumulation_steps not supported in CrossEncoder.fit
    )

    # Ensure full model artifacts are written to output_path
    os.makedirs(output_path, exist_ok=True)
    try:
        model.save(output_path)
    except Exception as exc:
        logger.warning(f"Failed to save Cross-Encoder model explicitly: {exc}")

    logger.info(f"Cross-Encoder saved → {output_path}")
    return model





# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str = "config.yaml"):
    from data_pipeline.utils import load_chunks, load_queries, load_qrels

    cfg = load_config(config_path)
    ce_cfg = cfg["cross_encoder_training"]

    if not ce_cfg["enabled"]:
        logger.info(
            "Cross-Encoder training disabled (cross_encoder_training.enabled=false).\n"
            "Using pretrained ms-marco-MiniLM-L-6-v2 — already strong on news domain.\n"
            "Set enabled=true in config.yaml to fine-tune."
        )
        return

    data_dir    = cfg["paths"]["data_dir"]
    output_path = ce_cfg["output_path"]

    logger.info("Loading preprocessed data …")
    chunks  = load_chunks(os.path.join(data_dir, "corpus_chunks.jsonl"))
    queries = load_queries(os.path.join(data_dir, "sampled_queries.json"))
    qrels   = load_qrels(os.path.join(data_dir, "sampled_qrels.json"))

    # ── Build Training Pairs ─────────────────────────────────────────────────
    cache_path = ce_cfg.get(
        "hard_neg_cache_path",
        os.path.join(cfg["paths"]["data_dir"], "ce_hard_negatives.jsonl"),
    )
    cached = load_ce_cache(cache_path)
    if cached:
        logger.info(f"Loaded cached CE pairs: {len(cached):,} samples")
        from sentence_transformers import InputExample
        examples = [InputExample(texts=[q, d], label=lbl) for q, d, lbl in cached]
    else:
        examples = build_ce_training_data(
            queries, qrels, chunks,
            top_k_neg=ce_cfg["hard_neg_top_k"],
            max_samples=ce_cfg["max_train_samples"],
            index_dir=cfg["paths"]["index_dir"],
            bi_encoder_model_name=cfg.get("bi_encoder_training", {}).get(
                "base_model", cfg.get("models", {}).get("bi_encoder_base")
            ),
        )
        if examples:
            save_ce_cache(cache_path, examples)
            logger.info(f"Cached CE pairs to: {cache_path}")

    if not examples:
        logger.error("No CE training examples built. Check qrels and BM25 index.")
        return

    # ── Evaluator ────────────────────────────────────────────────────────────
    evaluator = build_ce_evaluator(examples)

    # ── Train ────────────────────────────────────────────────────────────────
    model = train_cross_encoder(examples, evaluator, cfg, output_path)



    logger.info("\n✅ Cross-Encoder training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
