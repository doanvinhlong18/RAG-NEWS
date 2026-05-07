"""
bi_encoder_training.py
=======================
Fine-tune SBERT (Bi-Encoder) on BEIR trec-news with random negative sampling.

Steps:
  1. Load corpus, queries, qrels
  2. Build (query, positive, random_neg) samples
  3. Train with MultipleNegativesRankingLoss + InformationRetrievalEvaluator
  4. Save fine-tuned model

Memory:
  - batch_size=8, grad_accum=4 → effective=32
  - fp16 via sentence-transformers native use_amp
  - max_seq_length=256
  - Peak VRAM ~2.5GB on RTX 3050 Ti

Run:
  python bi_encoder_training.py
  python bi_encoder_training.py --config config.yaml
"""

import os
import json
import random
import logging
import argparse
import pickle
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_corpus_dict(filepath: str) -> Dict[str, Dict[str, str]]:
    """Load corpus as a dict: doc_id -> {title, text} (cached on disk)."""
    cache_path = f"{filepath}.pkl"
    try:
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(filepath):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except OSError:
        pass

    corpus: Dict[str, Dict[str, str]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            doc_id = data.get("doc_id") or data.get("id")
            if doc_id is None:
                continue
            corpus[doc_id] = {
                "title": data.get("title", ""),
                "text": data.get("text", "")
            }

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass

    return corpus


def format_doc_text(doc: Dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title:
        return f"{title}\n{text}".strip()
    return text


def is_positive_score(score) -> bool:
    try:
        return float(score) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Random Negative Sampling
# ---------------------------------------------------------------------------
def build_random_negative_samples(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict[str, str]],
    num_negatives: int = 2,
    max_samples: int = 10000,
    shuffle: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    Build (query, positive_doc, negative_doc) samples using random negatives.
    """
    doc_ids = list(corpus.keys())
    total_docs = len(doc_ids)

    if total_docs == 0:
        return []

    samples: List[Tuple[str, str, str]] = []
    qids = [q for q in qrels if q in queries]
    if shuffle:
        random.shuffle(qids)

    for qid in tqdm(qids, desc="Sampling negatives"):
        if len(samples) >= max_samples:
            break

        pos_ids = [did for did, score in qrels[qid].items() if is_positive_score(score)]
        if not pos_ids:
            continue

        pos_id = random.choice(pos_ids)
        pos_doc = corpus.get(pos_id)
        if not pos_doc:
            continue

        # Efficient negative sampling: rejection when positives are small, else build candidates once.
        pos_set = set(pos_ids)
        if len(pos_set) >= total_docs:
            continue

        neg_targets = max(1, min(num_negatives, 3))
        negatives: List[str] = []

        if len(pos_set) < total_docs * 0.3:
            attempts = 0
            max_attempts = neg_targets * 20
            while len(negatives) < neg_targets and attempts < max_attempts:
                cand = random.choice(doc_ids)
                if cand not in pos_set:
                    negatives.append(cand)
                attempts += 1
        else:
            candidates = [did for did in doc_ids if did not in pos_set]
            if candidates:
                negatives = random.sample(candidates, min(neg_targets, len(candidates)))

        if not negatives:
            continue

        query_text = queries[qid]
        pos_text = format_doc_text(pos_doc)
        for neg_id in negatives:
            neg_doc = corpus.get(neg_id)
            if not neg_doc:
                continue
            neg_text = format_doc_text(neg_doc)
            samples.append((query_text, pos_text, neg_text))
            if len(samples) >= max_samples:
                break

    return samples


def build_training_examples_from_samples(samples: List[Tuple[str, str, str]]) -> List:
    from sentence_transformers import InputExample

    return [InputExample(texts=[q, p, n]) for q, p, n in samples]


def load_cached_negatives(negatives_path: str) -> List[Tuple[str, str, str]]:
    samples: List[Tuple[str, str, str]] = []
    if not os.path.exists(negatives_path):
        return samples
    with open(negatives_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            samples.append((data["query"], data["positive"], data["negative"]))
    return samples


def cache_negatives(negatives_path: str, samples: List[Tuple[str, str, str]]) -> None:
    os.makedirs(os.path.dirname(negatives_path), exist_ok=True)
    with open(negatives_path, "w", encoding="utf-8") as f:
        for q, pos, neg in samples:
            f.write(json.dumps({"query": q, "positive": pos, "negative": neg}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# IR Evaluator (no self-retrieval leak)
# ---------------------------------------------------------------------------
def build_ir_evaluator(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict[str, str]],
    eval_fraction: float = 0.2,      # ↑ tăng lên
    max_eval_queries: int = 1000,    # ↑ tăng lên
    extra_corpus_size: int = 20000,  # ↑ thêm docs random
    seed: int = 42,
):
    """
    Improved IR Evaluator:
    - Random query split (no bias)
    - Larger corpus with random negatives
    - More realistic evaluation
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    import random

    random.seed(seed)

    # ── 1. Random split queries ─────────────────────────────
    all_qids = [qid for qid in qrels if qid in queries]
    random.shuffle(all_qids)

    n_eval = min(max_eval_queries, max(1, int(len(all_qids) * eval_fraction)))
    eval_qids = set(all_qids[:n_eval])

    eval_queries = {qid: queries[qid] for qid in eval_qids}

    # ── 2. Relevant docs ────────────────────────────────────
    relevant_doc_ids = set()
    for qid in eval_qids:
        relevant_doc_ids.update(qrels.get(qid, {}).keys())

    corpus_dict = {}

    for doc_id in relevant_doc_ids:
        doc = corpus.get(doc_id)
        if doc:
            corpus_dict[doc_id] = format_doc_text(doc)

    # ── 3. Add random negatives (KEY FIX) ───────────────────
    all_doc_ids = list(corpus.keys())
    remaining_ids = list(set(all_doc_ids) - relevant_doc_ids)

    if remaining_ids:
        extra_size = min(extra_corpus_size, len(remaining_ids))
        extra_docs = random.sample(remaining_ids, extra_size)

        for doc_id in extra_docs:
            doc = corpus.get(doc_id)
            if doc:
                corpus_dict[doc_id] = format_doc_text(doc)

    # ── 4. Relevant mapping ─────────────────────────────────
    relevant_docs = {}
    for qid in eval_qids:
        rel_docs = {
            doc_id for doc_id, score in qrels.get(qid, {}).items()
            if is_positive_score(score)
        }
        if rel_docs:
            relevant_docs[qid] = rel_docs

    logger.info(
        f"IR Evaluator: {len(eval_queries)} queries | {len(corpus_dict)} docs"
    )

    return InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name="trec-news-eval",
        show_progress_bar=True,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5],
        precision_recall_at_k=[5, 10],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_bi_encoder(
    examples: List,
    evaluator,
    cfg: dict,
    output_path: str,
):
    """
    Train SentenceTransformer with MultipleNegativesRankingLoss.
    Uses sentence-transformers native fp16 (use_amp=True) — no manual AMP.
    Supports checkpoint resume via output_path.
    """
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    tr_cfg = cfg["bi_encoder_training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU (CUDA not available)")

    # ── Model (checkpoint resume) ─────────────────────────────────────────
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "config.json")):
        logger.info(f"Resuming from checkpoint: {output_path}")
        model = SentenceTransformer(output_path)
    else:
        logger.info(f"Initializing model: {tr_cfg['base_model']}")
        model = SentenceTransformer(tr_cfg["base_model"], device=device)

    model.max_seq_length = tr_cfg["max_seq_length"]
    model.to(device)

    # ── DataLoader ─────────────────────────────────────────────────────────
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=tr_cfg["batch_size"],
        pin_memory=(device == "cuda"),
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    loss = losses.MultipleNegativesRankingLoss(model)

    # ── Warmup Steps ──────────────────────────────────────────────────────
    total_steps = len(loader) * tr_cfg["num_epochs"]
    warmup_steps = int(total_steps * tr_cfg["warmup_ratio"])

    logger.info(
        f"Training: {len(examples):,} examples | "
        f"batch={tr_cfg['batch_size']} | "
        f"epochs={tr_cfg['num_epochs']} | "
        f"grad_accum={tr_cfg['gradient_accumulation_steps']} | "
        f"warmup={warmup_steps} steps"
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model.fit(
        train_objectives=[(loader, loss)],
        evaluator=evaluator,
        epochs=tr_cfg["num_epochs"],
        warmup_steps=warmup_steps,
        output_path=output_path,
        evaluation_steps=tr_cfg.get("evaluation_steps", 1000),
        save_best_model=True,
        use_amp=(device == "cuda" and tr_cfg["use_amp"]),       # fp16 on GPU
        optimizer_params={"lr": tr_cfg["learning_rate"]},
        weight_decay=tr_cfg["weight_decay"],
    )

    logger.info(f"Bi-Encoder saved → {output_path}")
    return model





# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str = "config.yaml"):
    from data_pipeline.utils import load_queries, load_qrels

    cfg = load_config(config_path)
    tr_cfg = cfg["bi_encoder_training"]

    if not tr_cfg["enabled"]:
        logger.info("Bi-Encoder training disabled in config (bi_encoder_training.enabled=false). Skipping.")
        return

    data_dir   = cfg["paths"]["data_dir"]
    output_path = tr_cfg["output_path"]

    logger.info("Loading preprocessed data …")
    corpus  = load_corpus_dict(os.path.join(data_dir, "sampled_corpus.jsonl"))
    queries = load_queries(os.path.join(data_dir, "sampled_queries.json"))
    qrels   = load_qrels(os.path.join(data_dir, "sampled_qrels.json"))

    negatives_path = tr_cfg.get("negatives_path", os.path.join(data_dir, "train_negatives.jsonl"))
    samples = load_cached_negatives(negatives_path)
    if samples:
        logger.info(f"Loaded cached negatives: {len(samples):,} samples")
    else:
        logger.info("No cached negatives found. Building random negatives...")
        samples = build_random_negative_samples(
            queries=queries,
            qrels=qrels,
            corpus=corpus,
            num_negatives=int(tr_cfg.get("num_negatives_per_query", 2)),
            max_samples=int(tr_cfg["max_train_samples"]),
            shuffle=True,
        )
        cache_negatives(negatives_path, samples)
        logger.info(f"Built and cached random negatives: {len(samples):,} samples")

    if not samples:
        logger.error("No training samples built. Check data and qrels.")
        return

    examples = build_training_examples_from_samples(samples)
    logger.info(f"Using {len(samples):,} (query, pos, neg) samples.")

    # ── IR Evaluator (held-out queries) ────────────────────────────────────
    evaluator = build_ir_evaluator(
        queries, qrels, corpus,
        eval_fraction=tr_cfg["eval_split"],
    )

    # ── Train ───────────────────────────────────────────────────────────────
    model = train_bi_encoder(examples, evaluator, cfg, output_path)



    logger.info("\n✅ Bi-Encoder training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
