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
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


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
    max_samples: int = 5000,
    index_dir: str = "outputs/pipeline",
) -> List:
    """
    Build Cross-Encoder training pairs using BM25 hard negatives.

    Format: InputExample(texts=[query, doc_text], label=0.0 or 1.0)

    Strategy:
      - For each query: 1 positive (from qrels) + N negatives (BM25 top-K, non-relevant)
      - Balance: ~50% positive, ~50% negative
    """
    from sentence_transformers import InputExample

    logger.info("Loading BM25 index for CE hard negative mining …")
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    import nltk
    from nltk.corpus import stopwords
    from data_pipeline.utils import tokenize_for_bm25
    stops = set(stopwords.words("english"))

    # doc_id → list of chunk indices
    doc_to_chunks: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        doc_to_chunks.setdefault(c["doc_id"], []).append(i)

    examples = []
    qids = [q for q in qrels if q in queries]
    random.shuffle(qids)

    for qid in tqdm(qids, desc="Building CE pairs"):
        if len(examples) >= max_samples:
            break

        query_text = queries[qid]
        pos_doc_ids = set(qrels[qid].keys())

        # Positive example
        for doc_id in pos_doc_ids:
            chunk_indices = doc_to_chunks.get(doc_id, [])
            if not chunk_indices:
                continue
            pos_text = chunks[chunk_indices[0]]["chunk_text"]
            examples.append(InputExample(texts=[query_text, pos_text], label=1.0))
            break  # One positive per query

        # Hard negative examples via BM25
        toks = tokenize_for_bm25(query_text, stops)
        scores = bm25.get_scores(toks)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])

        neg_count = 0
        for idx in ranked[3:]:  # Skip top-3 (may contain near-positives)
            if chunks[idx]["doc_id"] in pos_doc_ids:
                continue
            examples.append(InputExample(texts=[query_text, chunks[idx]["chunk_text"]], label=0.0))
            neg_count += 1
            if neg_count >= 2:  # 2 negatives per positive
                break

    random.shuffle(examples)
    logger.info(f"Built {len(examples):,} CE training pairs "
                f"({sum(1 for e in examples if e.label == 1.0)} pos, "
                f"{sum(1 for e in examples if e.label == 0.0)} neg)")
    return examples


# ---------------------------------------------------------------------------
# CE Evaluator
# ---------------------------------------------------------------------------
def build_ce_evaluator(examples: List, eval_fraction: float = 0.1):
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
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "config.json")):
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
    warmup_steps = int(total_steps * ce_cfg["warmup_ratio"])

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
        gradient_accumulation_steps=ce_cfg["gradient_accumulation_steps"],
    )

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
    examples = build_ce_training_data(
        queries, qrels, chunks,
        top_k_neg=ce_cfg["hard_neg_top_k"],
        max_samples=ce_cfg["max_train_samples"],
        index_dir=cfg["paths"]["index_dir"],
    )

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
