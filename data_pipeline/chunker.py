import os
import json
import logging
import re
from typing import Iterator, Dict

logger = logging.getLogger(__name__)

def word_tokenize_simple(text: str):
    """
    Simple word tokenization to avoid loading heavy NLTK models into memory
    when processing millions of docs.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if w.strip()]

def chunk_document_stream(text: str, max_tokens: int, overlap: int) -> Iterator[Dict]:
    tokens = word_tokenize_simple(text)
    if not tokens:
        return
        
    step = max_tokens - overlap
    if step <= 0:
        step = max_tokens
        
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = " ".join(chunk_tokens)
        yield {
            'chunk_text': chunk_text,
            'n_tokens': len(chunk_tokens),
            'chunk_tokens': chunk_tokens
        }

def process_corpus_streaming(output_dir: str, max_tokens: int = 256, overlap: int = 32):
    """
    Reads sampled_corpus.jsonl line-by-line, chunks it, and writes to corpus_chunks.jsonl.
    Zero large list aggregations in memory.
    """
    logger.info(f"Chunking corpus (streaming) with max_tokens={max_tokens}, overlap={overlap}...")
    
    input_path = os.path.join(output_dir, "sampled_corpus.jsonl")
    output_path = os.path.join(output_dir, "corpus_chunks.jsonl")
    
    total_chunks = 0
    total_docs = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
         
        for line in f_in:
            if not line.strip():
                continue
                
            doc = json.loads(line)
            doc_id = doc['doc_id']
            text = doc.get('text', '')
            title = doc.get('title', '')
            
            # Combine title and text for chunking
            full_text = f"{title}. {text}" if title else text
            
            chunk_idx = 0
            for chunk in chunk_document_stream(full_text, max_tokens, overlap):
                chunk_id = f"{doc_id}__chunk_{chunk_idx}"
                out_record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk['chunk_text'],
                    "n_tokens": chunk['n_tokens'],
                    "chunk_tokens": chunk['chunk_tokens'] # Keep tokens for BM25 stream building
                }
                f_out.write(json.dumps(out_record) + "\n")
                chunk_idx += 1
                total_chunks += 1
                
            total_docs += 1
            
            if total_docs % 10000 == 0:
                logger.info(f"Processed {total_docs} docs, generated {total_chunks} chunks...")
                
    logger.info(f"Chunking complete. Total docs: {total_docs}, Total chunks: {total_chunks}")
