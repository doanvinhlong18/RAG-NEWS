import argparse
import logging
import os
import json

from data_pipeline.data_loader import extract_metadata, save_sampled_data
from data_pipeline.sampler import stratified_sample_metadata
from data_pipeline.splitter import split_data
from data_pipeline.chunker import process_corpus_streaming
from data_pipeline.index_builder import build_bm25_index, build_faiss_index
from data_pipeline.training_dataset_builder import build_training_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Memory-Safe RAG Data Pipeline")
    parser.add_argument("--repo_id", default="BeIR/trec-news", help="HuggingFace dataset repo")
    parser.add_argument("--dataset_name", default="trec-news", help="Dataset name config")
    parser.add_argument("--output_dir", default="outputs/pipeline", help="Directory for all outputs")
    parser.add_argument("--ratio", type=float, default=0.5, help="Sampling ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder model")
    parser.add_argument("--build-training-datasets", action="store_true", help="Also generate training triplets for bi-/cross-encoder training")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Pass 1: Extract Metadata (Stream) & 2. Sampling & 3. Save Sampled Data
    sampled_corpus_path = os.path.join(args.output_dir, "sampled_corpus.jsonl")
    if not os.path.exists(sampled_corpus_path):
        corpus_meta, queries_meta, qrels_dict = extract_metadata(args.repo_id, args.dataset_name)
        
        logger.info(f"Original Stats -> Corpus: {len(corpus_meta)}, Queries: {len(queries_meta)}")
        
        sampled_qids, sampled_doc_ids = stratified_sample_metadata(
            corpus_meta, queries_meta, qrels_dict, args.ratio, args.seed
        )
        
        logger.info(f"Sampled Stats -> Corpus: {len(sampled_doc_ids)}, Queries: {len(sampled_qids)}")
        
        save_sampled_data(
            args.repo_id, args.dataset_name, args.output_dir, 
            sampled_qids, sampled_doc_ids, qrels_dict
        )
        
        # Save Pipeline Stats
        stats = {
            "original_corpus_size": len(corpus_meta),
            "sampled_corpus_size": len(sampled_doc_ids),
            "original_queries_size": len(queries_meta),
            "sampled_queries_size": len(sampled_qids),
            "ratio": args.ratio
        }
        with open(os.path.join(args.output_dir, "pipeline_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)
    else:
        logger.info(f"Skipping extraction & sampling, found {sampled_corpus_path}")
    
    # 4. Split Queries
    train_queries_path = os.path.join(args.output_dir, "train_queries.json")
    if not os.path.exists(train_queries_path):
        split_data(args.output_dir)
    else:
        logger.info(f"Skipping query splitting, found {train_queries_path}")
    
    # 5. Chunking & Tokenizing
    chunks_path = os.path.join(args.output_dir, "corpus_chunks.jsonl")
    if not os.path.exists(chunks_path):
        process_corpus_streaming(args.output_dir)
    else:
        logger.info(f"Skipping chunking & tokenizing, found {chunks_path}")
    
    # 6. Indexing
    bm25_path = os.path.join(args.output_dir, "bm25.pkl")
    faiss_path = os.path.join(args.output_dir, "faiss.index")
    
    if not os.path.exists(bm25_path):
        build_bm25_index(args.output_dir)
    else:
        logger.info(f"Skipping BM25 indexing, found {bm25_path}")
        
    if not os.path.exists(faiss_path):
        build_faiss_index(args.output_dir, args.model_name)
    else:
        logger.info(f"Skipping FAISS indexing, found {faiss_path}")
    
    # 7. Training Datasets (optional; heavy and only needed for model training)
    if args.build_training_datasets:
        train_dataset_path = os.path.join(args.output_dir, "train.jsonl")
        if not os.path.exists(train_dataset_path):
            logger.info("Building optional training datasets because --build-training-datasets was requested.")
            build_training_datasets(args.output_dir)
        else:
            logger.info(f"Skipping training datasets builder, found {train_dataset_path}")
    else:
        logger.info("Skipping optional training dataset generation for faster demo/index builds.")
        
    logger.info("Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    main()
