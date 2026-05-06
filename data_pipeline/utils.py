import os
import json
import re
from typing import Dict, List, Set

def word_tokenize_simple(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if w.strip()]

def tokenize_for_bm25(text: str, stops: Set[str]) -> List[str]:
    tokens = word_tokenize_simple(text)
    return [t for t in tokens if t not in stops]

def load_chunks(filepath: str) -> Dict[str, str]:
    """Returns dict of chunk_id -> chunk_text"""
    chunks = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            chunks[data['chunk_id']] = data['chunk_text']
    return chunks

def load_queries(filepath: str) -> Dict[str, str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_qrels(filepath: str) -> Dict[str, Dict[str, int]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_chunk_to_doc_map(filepath: str) -> Dict[str, str]:
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            mapping[data['chunk_id']] = data['doc_id']
    return mapping

def load_bm25_tokenized(filepath: str) -> Dict[str, List[str]]:
    """Loads pre-tokenized chunks for BM25. chunk_id -> list of tokens."""
    tokenized = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            tokenized[data['chunk_id']] = data.get('chunk_tokens', [])
    return tokenized
