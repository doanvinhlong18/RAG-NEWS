import logging
import random
import math
from collections import defaultdict
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)

def get_quantiles(values, q=3):
    if not values:
        return []
    sorted_vals = sorted(values)
    quantiles = []
    for i in range(1, q):
        idx = int(len(sorted_vals) * i / q)
        quantiles.append(sorted_vals[idx])
    return quantiles

def get_bin_index(value, quantiles):
    for i, q_val in enumerate(quantiles):
        if value <= q_val:
            return i
    return len(quantiles)

def stratified_sample_metadata(
    corpus_meta: Dict[str, int], 
    queries_meta: Dict[str, int], 
    qrels_dict: Dict[str, Dict[str, int]], 
    ratio: float = 0.5, 
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Stratified sampling based ONLY on metadata (lengths, qrel counts).
    corpus_meta: {doc_id: length_in_words}
    queries_meta: {qid: length_in_words}
    qrels_dict: {qid: {doc_id: score}}
    """
    logger.info(f"Starting stratified metadata sampling with ratio={ratio}, seed={seed}")
    random.seed(seed)
    
    num_original_docs = len(corpus_meta)
    num_original_queries = len(queries_meta)
    
    # Corpus length distribution
    all_doc_lengths = list(corpus_meta.values())
    doc_len_quantiles = get_quantiles(all_doc_lengths, q=3)
    logger.info(f"Corpus length quantiles: {doc_len_quantiles}")
    
    original_corpus_bins = defaultdict(int)
    for length in all_doc_lengths:
        bin_idx = get_bin_index(length, doc_len_quantiles)
        original_corpus_bins[bin_idx] += 1
        
    # Query Stratification
    query_features = {}
    for qid in queries_meta.keys():
        q_qrels = qrels_dict.get(qid, {})
        q_num_qrels = len(q_qrels)
        
        rel_lengths = [corpus_meta.get(did, 0) for did in q_qrels.keys() if did in corpus_meta]
        q_avg_doc_len = sum(rel_lengths) / len(rel_lengths) if rel_lengths else 0
        
        query_features[qid] = {
            'num_qrels': q_num_qrels,
            'avg_doc_len': q_avg_doc_len
        }
        
    qrels_counts = [f['num_qrels'] for f in query_features.values()]
    q_doc_lens = [f['avg_doc_len'] for f in query_features.values()]
    
    qrels_quantiles = get_quantiles(qrels_counts, q=3)
    q_doc_len_quantiles = get_quantiles(q_doc_lens, q=3)
    
    query_strata = defaultdict(list)
    for qid, feats in query_features.items():
        bin_qrels = get_bin_index(feats['num_qrels'], qrels_quantiles)
        bin_doc_len = get_bin_index(feats['avg_doc_len'], q_doc_len_quantiles)
        stratum_key = f"{bin_qrels}_{bin_doc_len}"
        query_strata[stratum_key].append(qid)
        
    # Sample queries
    sampled_query_ids = set()
    for stratum_key, qids in query_strata.items():
        sample_size = math.ceil(len(qids) * ratio)
        sampled_qids = random.sample(qids, min(sample_size, len(qids)))
        sampled_query_ids.update(sampled_qids)
        
    logger.info(f"Sampled {len(sampled_query_ids)} queries out of {num_original_queries}")
    
    # Required Documents Collection
    sampled_doc_ids = set()
    for qid in sampled_query_ids:
        if qid in qrels_dict:
            for did in qrels_dict[qid].keys():
                if did in corpus_meta:
                    sampled_doc_ids.add(did)
                    
    logger.info(f"Kept {len(sampled_doc_ids)} relevant docs from sampled queries")
    
    # Fill Remaining Corpus
    target_corpus_size = math.ceil(num_original_docs * ratio)
    docs_needed = target_corpus_size - len(sampled_doc_ids)
    
    if docs_needed > 0:
        logger.info(f"Need {docs_needed} more docs to reach {ratio} ratio")
        current_sampled_bins = defaultdict(int)
        for did in sampled_doc_ids:
            bin_idx = get_bin_index(corpus_meta[did], doc_len_quantiles)
            current_sampled_bins[bin_idx] += 1
            
        target_bins = {
            bin_idx: math.ceil(count * ratio) 
            for bin_idx, count in original_corpus_bins.items()
        }
        
        unrelated_docs = set(corpus_meta.keys()) - sampled_doc_ids
        unrelated_by_bin = defaultdict(list)
        for did in unrelated_docs:
            bin_idx = get_bin_index(corpus_meta[did], doc_len_quantiles)
            unrelated_by_bin[bin_idx].append(did)
            
        for bin_idx, target_count in target_bins.items():
            current_count = current_sampled_bins.get(bin_idx, 0)
            needed_in_bin = target_count - current_count
            if needed_in_bin > 0 and bin_idx in unrelated_by_bin:
                available_in_bin = unrelated_by_bin[bin_idx]
                sample_size = min(needed_in_bin, len(available_in_bin))
                sampled_extras = random.sample(available_in_bin, sample_size)
                sampled_doc_ids.update(sampled_extras)
                
    logger.info(f"Final sampled corpus size: {len(sampled_doc_ids)}")
    return sampled_query_ids, sampled_doc_ids
