"""
cross_encoder_training.py
====================================
Fine-tune Cross-Encoder (MiniLM-L6-v2) on BEIR trec-news with hard negatives.

Root-cause fixes applied:
  [FIX-1] LambdaLoss (listwise + pairwise): each query is 1 training sample
          with docs=[pos, neg1..negN] and labels=teacher_scores[0..N].
          LambdaLoss applies pairwise ranking loss over ALL (doc_i, doc_j)
          pairs within the group, weighted by |ΔNDCG| — so mis-ordering a
          highly relevant doc costs more than mis-ordering a marginal one.
          Teacher scores each doc ONCE per query (not once per triplet),
          cutting scoring cost from 2×N to (1+N) pairs per query.
  [FIX-2] CERerankingEvaluator: replace CEBinaryClassificationEvaluator
          (measures AP/F1) with CERerankingEvaluator (measures MRR@10),
          so early stopping and model selection use the correct ranking metric.
  [FIX-3] Mine hard negatives using the FINETUNED bi-encoder (falls back to
          base if not present). Training negatives now match the inference
          distribution that CE will actually see at evaluation time.
  [FIX-4] Stratified negative sampling across hard (rank 2-12), medium
          (rank 12-32), and easy (rank 32-52) zones so CE learns to
          discriminate at all difficulty levels.
  [FIX-5] Freeze first N transformer layers + lower LR (5e-6) + fewer epochs
          (2) to prevent catastrophic forgetting of MS MARCO knowledge.
  [FIX-A] SigmoidCrossEncoder wrapper: sigmoid applied explicitly in predict()
          at both train and inference time, surviving save/load cycles.
  [FIX-B] Custom PyTorch training loop — bypasses sentence-transformers
          Trainer API to avoid 'unexpected keyword argument prompt' error
          introduced in newer library versions.
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


def load_ce_cache(cache_path: str) -> List[Dict]:
    """Load cached groups: list of {query, docs, labels}."""
    if not os.path.exists(cache_path):
        return []
    groups = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            groups.append(json.loads(line))
    return groups


def save_ce_cache(cache_path: str, groups: List[Dict]) -> None:
    """Save groups: each line = {query, docs, labels}."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# FAISS resource loader
# ---------------------------------------------------------------------------

def load_faiss_resources(
    index_dir: str,
) -> Tuple[Any, List[Tuple[str, str]]]:
    import faiss

    index_path   = os.path.join(index_dir, "faiss.index")
    meta_path    = os.path.join(index_dir, "chunk_metadata.pkl")
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
# Layer freezing helper  [FIX-5]
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
            logger.warning("Layer freezing: could not find bert/roberta/distilbert — skipping.")
            return

        for param in encoder.embeddings.parameters():
            param.requires_grad = False

        layers = encoder.encoder.layer
        for i in range(min(n_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

        frozen = sum(1 for p in auto_model.parameters() if not p.requires_grad)
        total  = sum(1 for p in auto_model.parameters())
        total_layers = len(layers)
        logger.info(
            "Froze %d / %d transformer layers (%d/%d param tensors frozen) — "
            "%d layers remain trainable",
            n_layers, total_layers, frozen, total, total_layers - n_layers,
        )
    except Exception as exc:
        logger.warning("Layer freezing failed: %s", exc)


# ---------------------------------------------------------------------------
# Training Data Construction  [FIX-1, FIX-3, FIX-4]
# ---------------------------------------------------------------------------

def build_ce_training_data(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    chunks,
    top_k_neg: int = 50,
    index_dir: str = "outputs/pipeline",
    bi_encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    neg_per_pos: int = 5,
    skip_top_k: int = 2,
    teacher_model_name: str = None,
    teacher_batch_size: int = 128,
) -> List[Dict]:
    """
    Build GROUP training examples for LambdaLoss.

    1 query = 1 sample:
        {
          "query":  str,
          "docs":   [pos_doc, neg1, neg2, ..., neg_N],
          "labels": [s_pos,   s1,   s2,   ..., s_N],   # teacher scores in [0,1]
        }

    Teacher scores each doc ONCE per query:
        cost = n_queries × (1 + neg_per_pos) pairs
        vs triplet approach: n_queries × neg_per_pos × 2 pairs

    LambdaLoss in train_cross_encoder() applies pairwise ranking loss over
    ALL (doc_i, doc_j) pairs within each group, weighted by |ΔNDCG|.
    """
    from sentence_transformers import SentenceTransformer
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

    logger.info("Encoding %d queries for hard negative search …", len(query_texts))
    q_embs = bi_encoder.encode(
        query_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    search_k = skip_top_k + top_k_neg
    logger.info("FAISS search top-%d …", search_k)
    _, indices = index.search(q_embs, search_k)

    # ── Stratified negative zones  [FIX-4] ───────────────────────────────
    hard_quota   = max(1, neg_per_pos // 2)
    medium_quota = max(1, (neg_per_pos - hard_quota) // 2)
    easy_quota   = neg_per_pos - hard_quota - medium_quota

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

    # ── Build groups ──────────────────────────────────────────────────────
    raw_groups: List[Dict] = []
    pos_found = 0

    for row_idx, qid in enumerate(tqdm(qids, desc="Building CE groups")):
        query_text  = queries[qid]
        pos_doc_ids = {str(did) for did in qrels[qid].keys()}

        pos_text = None
        for doc_id in pos_doc_ids:
            pos_text = doc_id_to_chunk_text.get(doc_id)
            if pos_text:
                break
        if not pos_text:
            continue
        pos_found += 1

        row_indices = indices[row_idx]

        def _collect_negs(start_rank, end_rank, quota):
            negs = []
            for rank in range(start_rank, min(end_rank, len(row_indices))):
                if len(negs) >= quota:
                    break
                doc_id_key, neg_text = _resolve_text(row_indices[rank])
                if not neg_text or doc_id_key in pos_doc_ids:
                    continue
                negs.append(neg_text)
            return negs

        neg_texts = (
            _collect_negs(skip_top_k,      skip_top_k + 10, hard_quota)
            + _collect_negs(skip_top_k + 10, skip_top_k + 30, medium_quota)
            + _collect_negs(skip_top_k + 30, search_k,        easy_quota)
        )

        if not neg_texts:
            continue

        # docs[0] = positive, docs[1:] = negatives — order preserved for logging
        raw_groups.append({
            "query":  query_text,
            "docs":   [pos_text] + neg_texts,
            "labels": None,   # filled after teacher scoring
        })

    logger.info(
        "Groups built: %d / %d queries had a positive",
        pos_found, len(qids),
    )

    # ── Teacher scoring: score every (query, doc) ONCE  [FIX-1] ──────────
    # Flatten → score all pairs in one pass → redistribute back to groups.
    # Each doc gets exactly one score; no repeated scoring of pos_doc.
    if teacher_model_name:
        from sentence_transformers.cross_encoder import CrossEncoder as _CE

        logger.info("Teacher CE: %s", teacher_model_name)
        teacher = _CE(teacher_model_name, max_length=512)

        all_pairs: List[Tuple[str, str]] = []
        group_sizes: List[int] = []
        for g in raw_groups:
            pairs_for_group = [(g["query"], doc) for doc in g["docs"]]
            all_pairs.extend(pairs_for_group)
            group_sizes.append(len(pairs_for_group))

        logger.info(
            "Teacher scoring %d (query, doc) pairs — %d queries × %d docs …",
            len(all_pairs), len(raw_groups), 1 + neg_per_pos,
        )
        logits = teacher.predict(
            all_pairs, batch_size=teacher_batch_size, show_progress_bar=True,
        )
        import numpy as np
        scores = 1.0 / (1.0 + np.exp(-logits.astype("float64")))  # sigmoid → [0,1]

        offset = 0
        for g, size in zip(raw_groups, group_sizes):
            g["labels"] = scores[offset: offset + size].tolist()
            offset += size

        pos_scores = np.array([g["labels"][0] for g in raw_groups])
        neg_scores = np.array([s for g in raw_groups for s in g["labels"][1:]])
        logger.info(
            "Teacher scores — pos mean: %.3f | neg mean: %.3f | "
            "pos beats all negs: %.1f%%",
            float(pos_scores.mean()),
            float(neg_scores.mean()),
            float(np.mean([
                g["labels"][0] > max(g["labels"][1:]) for g in raw_groups
            ]) * 100),
        )
    else:
        logger.warning(
            "No teacher model — using hard labels: pos=1.0, neg=0.0. "
            "Set cross_encoder_base in config for soft distillation."
        )
        for g in raw_groups:
            g["labels"] = [1.0] + [0.0] * (len(g["docs"]) - 1)

    random.shuffle(raw_groups)
    logger.info(
        "CE training data: %d groups | %d docs/group (1 pos + %d neg)",
        len(raw_groups), 1 + neg_per_pos, neg_per_pos,
    )
    return raw_groups


# ---------------------------------------------------------------------------
# Evaluator  [FIX-2]: CERerankingEvaluator → MRR@10, not AP/F1
# ---------------------------------------------------------------------------

def build_ce_evaluator(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    doc_id_to_chunk_text: Dict[str, str],
    faiss_doc_ids: List[Tuple[str, str]],
    indices: Any,           # shape [len(qids), search_k] — row order MUST match qids
    qids: List[str],        # same list & order used when encoding for FAISS search
    skip_top_k: int = 2,
    top_k_neg: int = 50,
    eval_queries: int = 200,
):
    """
    Build CrossEncoderRerankingEvaluator samples (MRR@10).

    CRITICAL: qids[i] must correspond to indices[i] — pass the exact list
    used to encode queries for FAISS, not a re-derived list from qrels/queries.
    """
    try:
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
        EvaluatorClass = CrossEncoderRerankingEvaluator
    except ImportError:
        from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
        EvaluatorClass = CERerankingEvaluator

    eval_qids = qids[:eval_queries]
    samples   = []

    for row_idx, qid in enumerate(eval_qids):
        if qid not in queries or qid not in qrels:
            continue

        query_text  = queries[qid]
        pos_doc_ids = {str(did) for did in qrels[qid].keys()}

        positives = [
            doc_id_to_chunk_text[did]
            for did in pos_doc_ids
            if did in doc_id_to_chunk_text
        ]
        if not positives:
            continue

        negatives = []
        search_k  = skip_top_k + top_k_neg
        for rank in range(skip_top_k, min(search_k, len(indices[row_idx]))):
            raw_idx = int(indices[row_idx][rank])
            if raw_idx < 0 or raw_idx >= len(faiss_doc_ids):
                continue
            doc_id, chunk_id = faiss_doc_ids[raw_idx]
            if doc_id is None:
                continue
            doc_id_key = str(doc_id)
            if doc_id_key in pos_doc_ids:
                continue
            txt = doc_id_to_chunk_text.get(doc_id_key)
            if txt:
                negatives.append(txt)
            if len(negatives) >= 20:
                break

        if not negatives:
            logger.debug("Eval query %s: no negatives found, skipping", qid)
            continue

        samples.append({
            "query":    query_text,
            "positive": positives,
            "negative": negatives,
        })

    logger.info(
        "CrossEncoderRerankingEvaluator: %d / %d eval queries have candidates",
        len(samples), len(eval_qids),
    )
    if len(samples) == 0:
        logger.warning(
            "No eval samples built. Diagnosing: "
            "doc_id_to_chunk_text size=%d, faiss_doc_ids size=%d, "
            "eval_qids=%d, indices shape=%s",
            len(doc_id_to_chunk_text),
            len(faiss_doc_ids),
            len(eval_qids),
            getattr(indices, "shape", "unknown"),
        )
        # Show a sample of what FAISS returned vs what we have in text map
        if len(eval_qids) > 0 and len(indices) > 0:
            sample_row = indices[0]
            covered = 0
            for rank, raw_idx in enumerate(sample_row[:10]):
                raw_idx = int(raw_idx)
                if 0 <= raw_idx < len(faiss_doc_ids):
                    doc_id, _ = faiss_doc_ids[raw_idx]
                    in_map = str(doc_id) in doc_id_to_chunk_text if doc_id else False
                    covered += int(in_map)
                    logger.warning(
                        "  rank %d: faiss_idx=%d doc_id=%s in_text_map=%s",
                        rank, raw_idx, doc_id, in_map,
                    )
            logger.warning("  First query: %d/10 FAISS results covered by text map", covered)
    return EvaluatorClass(samples, name="ce-reranking-eval")


# ---------------------------------------------------------------------------
# SigmoidCrossEncoder  [FIX-A]
# ---------------------------------------------------------------------------

class SigmoidCrossEncoder:
    """
    Wrapper around CrossEncoder.
    - predict(): always applies sigmoid explicitly (survives save/load)
    - tokenize() / forward(): used by custom training loop [FIX-B]
    """

    def __init__(self, model_name_or_path: str, max_length: int = 256, device: str = None):
        from sentence_transformers.cross_encoder import CrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._ce = CrossEncoder(
            model_name_or_path,
            num_labels=1,
            max_length=max_length,
            device=device,
        )
        self._device     = device
        self._max_length = max_length

    def save(self, path: str):
        self._ce.save(path)

    def predict(self, pairs, show_progress_bar: bool = False, batch_size: int = 64):
        import numpy as np
        logits = self._ce.predict(
            pairs,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )
        return 1.0 / (1.0 + np.exp(-logits.astype("float64")))

    def tokenize(self, pairs):
        return self._ce.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self._ce.model(**kwargs)
        return out.logits.squeeze(-1)   # (n_docs,)

    @classmethod
    def load(cls, model_path: str, max_length: int = 256, device: str = None):
        from sentence_transformers.cross_encoder import CrossEncoder

        obj = cls.__new__(cls)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        obj._ce         = CrossEncoder(model_path, num_labels=1, max_length=max_length, device=device)
        obj._device     = device
        obj._max_length = max_length
        return obj


# ---------------------------------------------------------------------------
# LambdaLoss  [FIX-1] — listwise loss with pairwise NDCG-weighted gradient
# ---------------------------------------------------------------------------

def lambda_loss(
    scores: torch.Tensor,   # (n_docs,)  raw logits for all docs in the group
    labels: torch.Tensor,   # (n_docs,)  teacher relevance scores in [0,1]
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    LambdaLoss: pairwise ranking loss weighted by |ΔNDCG|.

    For every pair (i, j) where label_i > label_j:
        loss_ij = log(1 + exp(score_j - score_i)) * |ΔNDCG_ij|

    |ΔNDCG_ij| = gain from correctly ordering i above j, so the model is
    penalised more for mis-ordering highly relevant documents.

    This is both listwise (sees all docs at once) and pairwise
    (loss is a sum over all pairs) — combining benefits of both.
    """
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Pairwise score diff: score_i - score_j  (n × n)
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)

    # Pairwise label diff: label_i - label_j
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)

    # Mask: only penalise pairs where i should rank above j
    pos_pairs = (label_diff > 0).float()

    # NDCG gain weights
    sorted_labels, _ = torch.sort(labels, descending=True)
    ideal_dcg = (sorted_labels / torch.log2(
        torch.arange(2, n + 2, dtype=torch.float32, device=scores.device)
    )).sum().clamp(min=eps)

    ranks     = torch.arange(1, n + 1, dtype=torch.float32, device=scores.device)
    discounts = 1.0 / torch.log2(ranks + 1)

    # |ΔNDCG| when swapping pair (i, j)
    delta_ndcg = (
        (discounts.unsqueeze(1) - discounts.unsqueeze(0)).abs()
        * label_diff.abs()
        / ideal_dcg
    ).clamp(min=0)

    # LambdaLoss: log-sigmoid loss weighted by NDCG gain
    pair_loss = torch.log1p(torch.exp(-score_diff)) * delta_ndcg * pos_pairs

    n_pairs = pos_pairs.sum().clamp(min=1)
    return pair_loss.sum() / n_pairs


# ---------------------------------------------------------------------------
# Dataset for group-based listwise training
# ---------------------------------------------------------------------------

class QueryGroupDataset(torch.utils.data.Dataset):
    """
    Each item is one query group from build_ce_training_data():
        {"query": str, "docs": [pos, neg1..negN], "labels": [s0..sN]}

    Passed directly to DataLoader with batch_size=1 — one query per step,
    all docs in that query scored together by LambdaLoss.
    """

    def __init__(self, groups: List[Dict]):
        self.items = groups

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        g = self.items[idx]
        return {
            "pairs":  [(g["query"], doc) for doc in g["docs"]],
            "labels": g["labels"],
        }


def _collate_query_groups(batch):
    return batch


# ---------------------------------------------------------------------------
# Training  [FIX-B] — custom PyTorch loop, LambdaLoss
# ---------------------------------------------------------------------------

def train_cross_encoder(
    groups: List[Dict],
    evaluator,
    cfg: dict,
    output_path: str,
):
    """
    Custom PyTorch training loop with LambdaLoss.

    Bypasses sentence-transformers Trainer/fit() entirely [FIX-B] to avoid
    the 'unexpected keyword argument prompt' error in newer versions.

    Each training step:
      1. Take one query group (all its docs)
      2. Tokenize all (query, doc) pairs
      3. Forward pass → logits for all docs
      4. LambdaLoss: pairwise NDCG-weighted loss over all (doc_i, doc_j) pairs
      5. Backprop + update

    Model selection: save best checkpoint by MRR@10 (CERerankingEvaluator).
    """
    from torch.optim import AdamW
    from torch.amp import GradScaler, autocast
    from transformers import get_linear_schedule_with_warmup

    ce_cfg = cfg["cross_encoder_training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_resuming = os.path.exists(output_path) and os.path.exists(
        os.path.join(output_path, "config.json")
    )

    if is_resuming:
        logger.info("Resuming from checkpoint: %s", output_path)
        model = SigmoidCrossEncoder.load(
            output_path, max_length=ce_cfg["max_seq_length"], device=str(device)
        )
    else:
        logger.info("Initialising CE model: %s", ce_cfg["base_model"])
        model = SigmoidCrossEncoder(
            ce_cfg["base_model"], max_length=ce_cfg["max_seq_length"], device=str(device)
        )

    # [FIX-5] Freeze bottom layers
    freeze_n = ce_cfg.get("freeze_first_n_layers", 0)
    if freeze_n > 0 and not is_resuming:
        _freeze_transformer_layers(model._ce.model, freeze_n)

    # Split train / eval groups
    n_eval        = max(50, int(len(groups) * 0.1))
    train_groups  = groups[:-n_eval]

    dataset    = QueryGroupDataset(train_groups)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=_collate_query_groups,
        num_workers=0,
    )

    num_epochs  = ce_cfg["num_epochs"]
    total_steps = len(dataloader) * num_epochs

    if is_resuming:
        warmup_steps = 0
        effective_lr = ce_cfg["learning_rate"] * 0.3
        logger.info("Resume mode: warmup=0, lr=%.2e", effective_lr)
    else:
        warmup_steps = int(total_steps * ce_cfg["warmup_ratio"])
        effective_lr = ce_cfg["learning_rate"]

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model._ce.model.parameters()),
        lr=effective_lr,
        eps=1e-8,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = ce_cfg.get("use_amp", False) and device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None

    logger.info(
        "CE Training (LambdaLoss): %d query groups | "
        "epochs=%d | warmup=%d | lr=%.2e | fp16=%s | device=%s",
        len(dataset), num_epochs, warmup_steps, effective_lr, use_amp, device,
    )

    best_score = -float("inf")
    best_ckpt  = os.path.join(output_path, "best")
    os.makedirs(best_ckpt, exist_ok=True)

    for epoch in range(num_epochs):
        model._ce.model.train()
        epoch_loss  = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            group  = batch[0]
            pairs  = group["pairs"]    # [(query, doc), ...]
            labels = group["labels"]   # [float, ...]

            if len(pairs) < 2:
                continue

            encoded      = model.tokenize(pairs)
            encoded      = {k: v.to(device) for k, v in encoded.items()}
            label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    logits = model.forward(**encoded)        # (n_docs,)
                    loss   = lambda_loss(logits, label_tensor)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model._ce.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model.forward(**encoded)
                loss   = lambda_loss(logits, label_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model._ce.model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss  += loss.item()
            epoch_steps += 1
            pbar.set_postfix(loss=f"{epoch_loss / epoch_steps:.4f}")

        avg_loss = epoch_loss / max(1, epoch_steps)
        logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, num_epochs, avg_loss)

        # Evaluate after each epoch
        if evaluator is not None and len(evaluator.samples) > 0:
            model._ce.model.eval()
            result = evaluator(model._ce, output_path=output_path)
            # CrossEncoderRerankingEvaluator returns a dict of metrics
            if isinstance(result, dict):
                score = result.get(
                    "ce-reranking-eval_mrr@10",
                    result.get("ce-reranking-eval_map", next(iter(result.values()), 0.0))
                )
                logger.info(
                    "Epoch %d eval — MRR@10: %.4f | MAP: %.4f | NDCG@10: %.4f",
                    epoch + 1,
                    result.get("ce-reranking-eval_mrr@10", 0.0),
                    result.get("ce-reranking-eval_map", 0.0),
                    result.get("ce-reranking-eval_ndcg@10", 0.0),
                )
            else:
                score = float(result)
                logger.info("Epoch %d eval MRR@10: %.4f", epoch + 1, score)
            if score > best_score:
                best_score = score
                model.save(best_ckpt)
                logger.info("  ↑ New best (%.4f) — saved to %s", best_score, best_ckpt)
        elif evaluator is not None:
            logger.warning(
                "Epoch %d: evaluator has 0 samples — saving checkpoint anyway.",
                epoch + 1,
            )
            model.save(best_ckpt)

    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    logger.info("Cross-Encoder saved → %s  (best: %s)", output_path, best_ckpt)
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
    logger.info("CE ranking check [%s]: relevant=%.4f  irrelevant=%.4f", status, rel, irr)
    return rel > irr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml"):
    from data_pipeline.utils import load_chunks, load_queries, load_qrels
    from sentence_transformers import SentenceTransformer
    import numpy as np

    cfg    = load_config(config_path)
    ce_cfg = cfg["cross_encoder_training"]

    if not ce_cfg["enabled"]:
        logger.info("Cross-Encoder training disabled. Set enabled=true to fine-tune.")
        return

    data_dir    = cfg["paths"]["data_dir"]
    output_path = ce_cfg["output_path"]

    logger.info("Loading preprocessed data …")
    chunks  = load_chunks(os.path.join(data_dir, "corpus_chunks.jsonl"))
    queries = load_queries(os.path.join(data_dir, "sampled_queries.json"))
    qrels   = load_qrels(os.path.join(data_dir, "sampled_qrels.json"))

    # Cap queries — this is the ONLY cap; build_ce_training_data processes all of them
    max_q = ce_cfg.get("max_train_queries", 5000)
    if max_q and len(queries) > max_q:
        sampled_qids = random.sample(list(qrels.keys()), min(max_q, len(qrels)))
        queries = {q: queries[q] for q in sampled_qids if q in queries}
        qrels   = {q: qrels[q]   for q in sampled_qids if q in qrels}
        logger.info("Capped to %d queries for CE training.", len(queries))

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

    use_teacher  = ce_cfg.get("use_teacher_scores", True)
    teacher_name = cfg["models"]["cross_encoder_base"] if use_teacher else None

    # Cache v4: group format (query, docs, labels) — incompatible with v3 triplets
    cache_path = ce_cfg.get(
        "hard_neg_cache_path",
        os.path.join(cfg["paths"]["data_dir"], "ce_hard_negatives_v4.jsonl"),
    )

    groups = load_ce_cache(cache_path)
    if groups:
        logger.info("Loaded cached CE groups: %d", len(groups))
    else:
        groups = build_ce_training_data(
            queries,
            qrels,
            chunks,
            top_k_neg=ce_cfg["hard_neg_top_k"],
            index_dir=cfg["paths"]["index_dir"],
            bi_encoder_model_name=bi_for_neg,
            neg_per_pos=ce_cfg.get("neg_per_pos", 5),
            skip_top_k=ce_cfg.get("skip_top_k", 2),
            teacher_model_name=teacher_name,
        )
        if groups:
            save_ce_cache(cache_path, groups)
            logger.info("Cached CE groups → %s", cache_path)

    if not groups:
        logger.error("No CE training groups. Check qrels and FAISS index.")
        return

    # Build evaluator
    logger.info("Building CERerankingEvaluator …")
    try:
        index, faiss_doc_ids = load_faiss_resources(cfg["paths"]["index_dir"])
    except FileNotFoundError as exc:
        logger.error("Cannot build evaluator: %s", exc)
        return

    bi_encoder = SentenceTransformer(bi_for_neg)

    # Build eval_qids as a STABLE LIST — same order used to encode and search FAISS.
    # Must NOT re-derive from dict iteration (qrels/queries) after this point,
    # because row_idx in eval_indices corresponds to position in this list.
    n_eval_q     = ce_cfg.get("eval_queries", 200)
    all_eval_q   = [q for q in queries if q in qrels]   # intersection, stable order
    eval_qids    = all_eval_q[:n_eval_q]
    eval_q_texts = [queries[qid] for qid in eval_qids]  # same order as eval_qids

    logger.info("Encoding %d eval queries for FAISS search …", len(eval_qids))
    eval_embs = bi_encoder.encode(
        eval_q_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    search_k = ce_cfg.get("skip_top_k", 2) + ce_cfg["hard_neg_top_k"]
    _, eval_indices = index.search(eval_embs, search_k)
    # eval_indices[i] corresponds to eval_qids[i] — order is preserved

    # Rebuild doc_id_to_chunk_text for evaluator by streaming corpus_chunks.jsonl.
    # Cannot use the already-loaded `chunks` dict (chunk_id→text) because
    # list(dict) yields keys only, not dicts — so we re-read the raw JSONL.
    chunks_path = os.path.join(data_dir, "corpus_chunks.jsonl")
    chunk_id_to_text: Dict[str, str] = {}
    doc_id_to_chunk_text: Dict[str, str] = {}
    logger.info("Rebuilding doc_id_to_chunk_text from %s …", chunks_path)
    with open(chunks_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            if not _line.strip():
                continue
            _c = json.loads(_line)
            cid = _c.get("chunk_id")
            txt = _c.get("chunk_text")
            did = _c.get("doc_id")
            if cid and txt:
                chunk_id_to_text[str(cid)] = txt
            if did and txt:
                doc_id_to_chunk_text.setdefault(str(did), txt)
    # Also fill via FAISS mapping in case chunk_id format differs
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
        "doc_id_to_chunk_text: %d doc_ids | chunk_id_to_text: %d chunk_ids",
        len(doc_id_to_chunk_text), len(chunk_id_to_text),
    )

    evaluator = build_ce_evaluator(
        queries=queries,
        qrels=qrels,
        doc_id_to_chunk_text=doc_id_to_chunk_text,
        faiss_doc_ids=faiss_doc_ids,
        indices=eval_indices,
        qids=eval_qids,
        skip_top_k=ce_cfg.get("skip_top_k", 2),
        top_k_neg=ce_cfg["hard_neg_top_k"],
        eval_queries=ce_cfg.get("eval_queries", 200),
    )

    train_cross_encoder(groups, evaluator, cfg, output_path)

    logger.info("Running post-training ranking verification …")
    verify_ce_ranking(output_path, max_length=ce_cfg["max_seq_length"])

    logger.info("\n✅ Cross-Encoder training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)