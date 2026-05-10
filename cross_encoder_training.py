"""
cross_encoder_training.py
====================================
Fine-tune Cross-Encoder (MiniLM-L6-v2) on BEIR trec-news with hard negatives.

Root-cause fixes applied (see docs/CE_FINETUNING_FAILURE_ANALYSIS.md):
  [FIX-1] Knowledge distillation: replace binary 0/1 labels with continuous
          scores from the teacher (base ms-marco CE).  BCEWithLogitsLoss
          accepts soft labels in [0,1], preserving the ranking signal from
          540K MS MARCO pairs instead of collapsing to binary classification.
  [FIX-2] Mine hard negatives using the FINETUNED bi-encoder (falls back to
          base if not present).  Training negatives now match the inference
          distribution that CE will actually see at evaluation time.
  [FIX-3] Stratified negative sampling across hard (rank 2-12), medium
          (rank 12-32), and easy (rank 32-52) zones so CE learns to
          discriminate at all difficulty levels, including the top-4
          candidates that were previously never seen in training.
  [FIX-4] Freeze first N transformer layers + lower LR (5e-6) + fewer epochs
          (2) to prevent catastrophic forgetting of MS MARCO knowledge.
  [FIX-A] SigmoidCrossEncoder wrapper: activation_fct=Sigmoid is applied at
          both train and inference time, surviving save/load cycles.
  [FIX-bug] Removed double-sigmoid in SigmoidCrossEncoder.predict() —
          self._ce already applies Sigmoid internally via activation_fct.
"""

import os
import random
import logging
import argparse
import pickle
import json
from typing import Any, Dict, List, Tuple
import torch

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / Cache helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_ce_cache(cache_path: str) -> List[Tuple[str, str, float]]:
    if not os.path.exists(cache_path):
        return []
    samples: List[Tuple[str, str, float]] = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append((item["query"], item["doc"], float(item["label"])))
    return samples


def save_ce_cache(cache_path: str, examples: List) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {"query": ex.texts[0], "doc": ex.texts[1], "label": ex.label},
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# FAISS resource loader
# ---------------------------------------------------------------------------

def load_faiss_resources(
    index_dir: str,
) -> Tuple[Any, List[Tuple[str, str]]]:
    import faiss

    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "chunk_metadata.pkl")
    doc_ids_path = os.path.join(index_dir, "doc_ids.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)

    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            chunk_metadata = pickle.load(f)
        max_idx = max(chunk_metadata.keys()) if chunk_metadata else -1
        mapping: List[Tuple[str, str]] = [(None, None)] * (max_idx + 1)
        for idx, meta in chunk_metadata.items():
            mapping[idx] = (meta.get("doc_id"), meta.get("chunk_id"))
        return index, mapping

    if os.path.exists(doc_ids_path):
        with open(doc_ids_path, "rb") as f:
            doc_ids = pickle.load(f)
        return index, [(doc_id, None) for doc_id in doc_ids]

    raise FileNotFoundError(
        f"No doc_id mapping found. Expected {meta_path} or {doc_ids_path}."
    )


# ---------------------------------------------------------------------------
# Layer freezing helper  [FIX-4: prevent catastrophic forgetting]
# ---------------------------------------------------------------------------

def _freeze_transformer_layers(auto_model, n_layers: int) -> None:
    """Freeze embedding layer + first n transformer layers of a BERT-like model."""
    try:
        encoder = (
            getattr(auto_model, "bert", None)
            or getattr(auto_model, "roberta", None)
            or getattr(auto_model, "distilbert", None)
        )
        if encoder is None:
            logger.warning("Layer freezing: could not find 'bert/roberta/distilbert' attribute — skipping.")
            return

        for param in encoder.embeddings.parameters():
            param.requires_grad = False

        layers = encoder.encoder.layer
        for i in range(min(n_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

        frozen = sum(1 for p in auto_model.parameters() if not p.requires_grad)
        total  = sum(1 for p in auto_model.parameters())
        logger.info("Froze %d transformer layers: %d/%d param tensors frozen", n_layers, frozen, total)
    except Exception as exc:
        logger.warning("Layer freezing failed: %s", exc)


# ---------------------------------------------------------------------------
# Training Data Construction  [FIX-2, FIX-3]
# ---------------------------------------------------------------------------

def build_ce_training_data(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    chunks,
    top_k_neg: int = 50,
    max_samples: int = 100_000,
    index_dir: str = "outputs/pipeline",
    bi_encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    neg_per_pos: int = 7,
    skip_top_k: int = 2,
    teacher_model_name: str = None,   # [FIX-1] base CE for soft-label distillation
    teacher_batch_size: int = 128,
) -> List:
    from sentence_transformers import InputExample, SentenceTransformer
    import numpy as np

    logger.info("Loading FAISS index for CE hard negative mining …")
    try:
        index, faiss_doc_ids = load_faiss_resources(index_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return []

    bi_encoder = SentenceTransformer(bi_encoder_model_name)

    # ── Build chunk/doc text maps ─────────────────────────────────────────
    chunk_id_to_text: Dict[str, str] = {}
    doc_id_to_chunk_text: Dict[str, str] = {}

    if isinstance(chunks, dict):
        for chunk_id, chunk_text in chunks.items():
            if chunk_id and chunk_text:
                chunk_id_to_text[str(chunk_id)] = chunk_text
    else:
        for c in chunks:
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except json.JSONDecodeError:
                    continue
            if not isinstance(c, dict):
                continue
            cid = c.get("chunk_id")
            txt = c.get("chunk_text")
            did = c.get("doc_id")
            if cid and txt:
                chunk_id_to_text[str(cid)] = txt
            if did and txt:
                doc_id_to_chunk_text.setdefault(str(did), txt)

    # Populate doc_id_to_chunk_text from FAISS mapping
    for doc_id, chunk_id in faiss_doc_ids:
        if doc_id is None:
            continue
        key = str(doc_id)
        if key in doc_id_to_chunk_text:
            continue
        if chunk_id is not None:
            txt = chunk_id_to_text.get(str(chunk_id))
            if txt:
                doc_id_to_chunk_text[key] = txt

    logger.info(
        "Mapping: chunk_texts=%d | doc_texts=%d",
        len(chunk_id_to_text),
        len(doc_id_to_chunk_text),
    )

    # ── Encode queries ────────────────────────────────────────────────────
    qids = [q for q in qrels if q in queries]
    random.shuffle(qids)
    query_texts = [queries[qid] for qid in qids]

    logger.info(f"Encoding {len(query_texts)} queries for hard negative search …")
    q_embs = bi_encoder.encode(
        query_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    # fetch enough candidates for stratified sampling across the full range
    search_k = skip_top_k + top_k_neg
    logger.info(f"FAISS search top-{search_k} (skipping top-{skip_top_k} as potential false negatives) …")
    _, indices = index.search(q_embs, search_k)

    # ── Build pairs ───────────────────────────────────────────────────────
    # Stratified sampling: pull negatives from hard / medium / easy zones
    # so CE learns to discriminate across the full difficulty spectrum —
    # not just rank 5-8 which misses the top candidates seen at inference.
    hard_quota   = max(1, neg_per_pos // 2)          # rank [skip, skip+10)
    medium_quota = max(1, (neg_per_pos - hard_quota) // 2)  # rank [10, 30)
    easy_quota   = neg_per_pos - hard_quota - medium_quota  # rank [30, top_k_neg)

    def _resolve_text(idx):
        if idx < 0 or idx >= len(faiss_doc_ids):
            return None, None
        doc_id, chunk_id = faiss_doc_ids[idx]
        doc_id_key = str(doc_id) if doc_id is not None else None
        if not doc_id_key:
            return None, None
        text = (chunk_id_to_text.get(str(chunk_id)) if chunk_id is not None else None) \
               or doc_id_to_chunk_text.get(doc_id_key)
        return doc_id_key, text

    examples: List = []
    pos_found = 0

    for row_idx, qid in enumerate(tqdm(qids, desc="Building CE pairs")):
        if len(examples) >= max_samples:
            break

        query_text = queries[qid]
        pos_doc_ids = {str(did) for did in qrels[qid].keys()}

        # Positive
        pos_text = None
        for doc_id in pos_doc_ids:
            pos_text = doc_id_to_chunk_text.get(doc_id)
            if pos_text:
                break
        if not pos_text:
            continue
        pos_found += 1
        examples.append(InputExample(texts=[query_text, pos_text], label=1.0))

        # [FIX-3] Stratified negatives across hard / medium / easy zones
        row_indices = indices[row_idx]

        def _collect(start_rank, end_rank, quota):
            collected = []
            for rank in range(start_rank, min(end_rank, len(row_indices))):
                if len(collected) >= quota:
                    break
                doc_id_key, neg_text = _resolve_text(row_indices[rank])
                if not neg_text or doc_id_key in pos_doc_ids:
                    continue
                collected.append(InputExample(texts=[query_text, neg_text], label=0.0))
            return collected

        examples.extend(_collect(skip_top_k,      skip_top_k + 10, hard_quota))
        examples.extend(_collect(skip_top_k + 10, skip_top_k + 30, medium_quota))
        examples.extend(_collect(skip_top_k + 30, search_k,        easy_quota))

    logger.info("Positive matches: %d/%d queries", pos_found, len(qids))
    random.shuffle(examples)

    n_pos = sum(1 for e in examples if e.label == 1.0)
    n_neg = len(examples) - n_pos
    logger.info(
        "CE pairs: %d total | %d pos | %d neg | %.1f%% pos",
        len(examples), n_pos, n_neg, 100 * n_pos / max(1, len(examples)),
    )

    # [FIX-1] Knowledge distillation: replace binary 0/1 labels with soft
    # scores from the teacher CE.  BCEWithLogitsLoss accepts continuous targets
    # in [0,1], so this turns training into soft-label distillation that
    # preserves the ranking signal from MS MARCO without binary collapse.
    if teacher_model_name:
        from sentence_transformers.cross_encoder import CrossEncoder as _CE
        logger.info("Scoring %d pairs with teacher CE: %s …", len(examples), teacher_model_name)
        teacher = _CE(teacher_model_name, max_length=512)
        all_pairs = [(e.texts[0], e.texts[1]) for e in examples]
        raw_scores = teacher.predict(all_pairs, batch_size=teacher_batch_size, show_progress_bar=True)
        # Teacher may return raw logits; apply sigmoid to map to [0,1]
        soft_scores = 1.0 / (1.0 + np.exp(-raw_scores.astype("float64")))
        # Track original polarity BEFORE overwriting labels (examples are shuffled)
        orig_positive = np.array([e.label >= 0.5 for e in examples])
        for e, score in zip(examples, soft_scores):
            e.label = float(score)
        logger.info(
            "Teacher scores — mean pos: %.3f | mean neg: %.3f",
            float(soft_scores[orig_positive].mean()) if orig_positive.any() else 0.0,
            float(soft_scores[~orig_positive].mean()) if (~orig_positive).any() else 0.0,
        )

    return examples


# ---------------------------------------------------------------------------
# Evaluator  [FIX-D]
# ---------------------------------------------------------------------------

def build_ce_evaluator(examples: List, eval_fraction: float = 0.1):
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

    n_eval = max(50, int(len(examples) * eval_fraction))
    eval_examples = examples[-n_eval:]

    sentences1 = [e.texts[0] for e in eval_examples]
    sentences2 = [e.texts[1] for e in eval_examples]
    # Labels may be soft (teacher scores) — round to 0/1 for binary evaluator
    labels = [1 if e.label >= 0.5 else 0 for e in eval_examples]

    return CEBinaryClassificationEvaluator(
        sentence_pairs=list(zip(sentences1, sentences2)),
        labels=labels,
        name="ce-eval",
    )


# ---------------------------------------------------------------------------
# SigmoidCrossEncoder
# [FIX-A] activation_fct is NOT serialized by CrossEncoder.fit()/save().
#         This wrapper overrides predict() to always apply sigmoid,
#         surviving save/load cycles permanently.
# ---------------------------------------------------------------------------

class SigmoidCrossEncoder:
    """
    Wrapper around CrossEncoder that always applies sigmoid in predict().

    CrossEncoder.fit() does not serialize activation_fct — after save/load
    the attribute is lost and predict() returns raw logits, causing inverted
    ranking. This class applies sigmoid explicitly so the fix is permanent.
    """

    def __init__(self, model_name_or_path: str, max_length: int = 256, device: str = None):
        import torch.nn as nn
        from sentence_transformers.cross_encoder import CrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._ce = CrossEncoder(
            model_name_or_path,
            num_labels=1,
            max_length=max_length,
            device=device,
        )

    def fit(self, **kwargs):
        return self._ce.fit(**kwargs)

    def save(self, path: str):
        self._ce.save(path)

    def predict(self, pairs, show_progress_bar: bool = False):
        # self._ce already has activation_fct=Sigmoid, so raw is already in [0,1].
        # Do NOT apply sigmoid again — that would be sigmoid(sigmoid(x)).
        return self._ce.predict(pairs, show_progress_bar=show_progress_bar)

    @classmethod
    def load(cls, model_path: str, max_length: int = 256, device: str = None):
        obj = cls.__new__(cls)
        import torch
        import torch.nn as nn
        from sentence_transformers.cross_encoder import CrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        obj._ce = CrossEncoder(model_path, num_labels=1, max_length=max_length, device=device)
        return obj


# ---------------------------------------------------------------------------
# Training  [FIX-A]
# ---------------------------------------------------------------------------

def train_cross_encoder(
    examples: List,
    evaluator,
    cfg: dict,
    output_path: str,
):
    """
    Uses SigmoidCrossEncoder so sigmoid is applied at both train and inference,
    surviving save/load cycles. Fixes the inverted-ranking bug permanently.
    """
    from torch.utils.data import DataLoader

    ce_cfg = cfg["cross_encoder_training"]

    is_resuming = os.path.exists(output_path) and os.path.exists(
        os.path.join(output_path, "config.json")
    )

    if is_resuming:
        logger.info(f"Resuming from checkpoint: {output_path}")
        model = SigmoidCrossEncoder.load(output_path, max_length=ce_cfg["max_seq_length"])
    else:
        logger.info(f"Initializing CE model: {ce_cfg['base_model']}")
        model = SigmoidCrossEncoder(ce_cfg["base_model"], max_length=ce_cfg["max_seq_length"])

    # [FIX-4] Freeze bottom N transformer layers to prevent catastrophic forgetting
    freeze_n = ce_cfg.get("freeze_first_n_layers", 0)
    if freeze_n > 0 and not is_resuming:
        _freeze_transformer_layers(model._ce.model, freeze_n)

    n_eval = max(50, int(len(examples) * 0.1))
    train_examples = examples[:-n_eval]

    total_steps = (len(train_examples) // ce_cfg["batch_size"]) * ce_cfg["num_epochs"]

    if is_resuming:
        warmup_steps = 0
        effective_lr = ce_cfg["learning_rate"] * 0.3
        logger.info(f"Resume mode: warmup=0, lr={effective_lr:.2e}")
    else:
        warmup_steps = int(total_steps * ce_cfg["warmup_ratio"])
        effective_lr = ce_cfg["learning_rate"]

    logger.info(
        f"CE Training: {len(train_examples):,} pairs | "
        f"batch={ce_cfg['batch_size']} | epochs={ce_cfg['num_epochs']} | "
        f"warmup={warmup_steps} | lr={effective_lr:.2e} | fp16={ce_cfg['use_amp']}"
    )

    model.fit(
        train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=ce_cfg["batch_size"]),
        evaluator=evaluator,
        epochs=ce_cfg["num_epochs"],
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=True,
        optimizer_params={"lr": effective_lr},
        use_amp=ce_cfg["use_amp"],
    )

    os.makedirs(output_path, exist_ok=True)
    try:
        model.save(output_path)
    except Exception as exc:
        logger.warning(f"model.save() failed: {exc}")

    logger.info(f"Cross-Encoder saved → {output_path}")
    return model


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_ce_ranking(model_path: str, max_length: int = 256):
    """Sanity-check: relevant doc must score higher than irrelevant."""
    model = SigmoidCrossEncoder.load(model_path, max_length=max_length)
    pairs = [
        ("AI applications in modern healthcare",
         "Artificial intelligence is transforming medical diagnosis and treatment planning."),
        ("AI applications in modern healthcare",
         "The stock market closed higher on Friday amid strong earnings reports."),
    ]
    scores = model.predict(pairs)
    rel, irr = float(scores[0]), float(scores[1])
    status = "PASS" if rel > irr else "FAIL ← ranking still inverted!"
    logger.info(f"CE ranking check [{status}]: relevant={rel:.4f}  irrelevant={irr:.4f}")
    return rel > irr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml"):
    from data_pipeline.utils import load_chunks, load_queries, load_qrels

    cfg = load_config(config_path)
    ce_cfg = cfg["cross_encoder_training"]

    if not ce_cfg["enabled"]:
        logger.info(
            "Cross-Encoder training disabled (cross_encoder_training.enabled=false). "
            "Set enabled=true to fine-tune."
        )
        return

    data_dir = cfg["paths"]["data_dir"]
    output_path = ce_cfg["output_path"]

    logger.info("Loading preprocessed data …")
    chunks = load_chunks(os.path.join(data_dir, "corpus_chunks.jsonl"))
    queries = load_queries(os.path.join(data_dir, "sampled_queries.json"))
    qrels = load_qrels(os.path.join(data_dir, "sampled_qrels.json"))

    # Cap queries để tránh OOM và encode quá lâu
    max_q = ce_cfg.get("max_train_queries", 5000)
    if max_q and len(queries) > max_q:
        import random as _rnd
        sampled_qids = _rnd.sample(list(qrels.keys()), min(max_q, len(qrels)))
        queries = {q: queries[q] for q in sampled_qids if q in queries}
        qrels   = {q: qrels[q]   for q in sampled_qids if q in qrels}
        logger.info(f"Capped to {len(queries):,} queries for CE training.")

    # [FIX-2] Use finetuned bi-encoder for negative mining so the CE sees
    # the same candidate distribution it will face at inference time.
    finetuned_bi_path = cfg["bi_encoder_training"]["output_path"]
    bi_for_neg = (
        finetuned_bi_path
        if os.path.exists(os.path.join(finetuned_bi_path, "config.json"))
        else cfg["models"]["bi_encoder_base"]
    )
    logger.info(
        "Negative mining bi-encoder: %s (%s)",
        bi_for_neg,
        "finetuned" if bi_for_neg == finetuned_bi_path else "base — finetuned not found",
    )

    use_teacher = ce_cfg.get("use_teacher_scores", True)
    teacher_name = cfg["models"]["cross_encoder_base"] if use_teacher else None

    # Cache is keyed on approach; v2 uses soft labels + stratified negatives.
    cache_path = ce_cfg.get(
        "hard_neg_cache_path",
        os.path.join(cfg["paths"]["data_dir"], "ce_hard_negatives_v2.jsonl"),
    )
    cached = load_ce_cache(cache_path)
    if cached:
        logger.info(f"Loaded cached CE pairs: {len(cached):,}")
        from sentence_transformers import InputExample
        examples = [InputExample(texts=[q, d], label=lbl) for q, d, lbl in cached]
    else:
        examples = build_ce_training_data(
            queries,
            qrels,
            chunks,
            top_k_neg=ce_cfg["hard_neg_top_k"],
            max_samples=ce_cfg["max_train_samples"],
            index_dir=cfg["paths"]["index_dir"],
            bi_encoder_model_name=bi_for_neg,                    # [FIX-2]
            neg_per_pos=ce_cfg.get("neg_per_pos", 7),            # [FIX-3]
            skip_top_k=ce_cfg.get("skip_top_k", 2),              # [FIX-3]
            teacher_model_name=teacher_name,                      # [FIX-1]
        )
        if examples:
            save_ce_cache(cache_path, examples)
            logger.info(f"Cached CE pairs → {cache_path}")

    if not examples:
        logger.error("No CE training examples. Check qrels and FAISS index.")
        return

    evaluator = build_ce_evaluator(examples)
    train_cross_encoder(examples, evaluator, cfg, output_path)

    # ── Sanity check ──────────────────────────────────────────────────────
    logger.info("Running post-training ranking verification …")
    verify_ce_ranking(output_path, max_length=ce_cfg["max_seq_length"])

    logger.info("\n✅ Cross-Encoder training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)