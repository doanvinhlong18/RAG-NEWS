import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

def identity(x):
    return x

class SparseBM25:
    """
    A memory-efficient implementation of BM25Okapi using scipy sparse matrices.
    Provides a compatible .get_scores() interface to rank_bm25 but avoids holding
    a massive list of dictionaries in memory, enabling serialization of huge datasets.
    """
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.vectorizer = CountVectorizer(
            analyzer=identity, 
            dtype=np.float32
        )
        
    def fit(self, tokenized_corpus_generator):
        # Fit and transform the corpus into a sparse matrix
        X = self.vectorizer.fit_transform(tokenized_corpus_generator)
        self.corpus_size = X.shape[0]
        
        # Document lengths
        self.doc_len = np.array(X.sum(axis=1)).flatten()
        self.avgdl = self.doc_len.mean()
        
        # Calculate IDF
        X_bool = X.copy()
        X_bool.data = np.ones_like(X_bool.data) # Set all counts to 1 to calculate doc frequency
        nd = np.array(X_bool.sum(axis=0)).flatten()
        
        # rank_bm25 exact IDF formula
        idf = np.log(self.corpus_size - nd + 0.5) - np.log(nd + 0.5)
        
        # Handle negative IDFs (standard rank_bm25 behavior)
        if np.any(idf > 0):
            avg_idf = np.sum(idf[idf > 0]) / max(np.sum(idf > 0), 1)
        else:
            avg_idf = 0.0
        eps_idf = self.epsilon * avg_idf
        idf[idf < 0] = eps_idf
        
        self.idf = idf
        
        # Convert to CSC format for fast column slicing during queries
        self.X_csc = X.tocsc()
        
    def get_scores(self, query):
        vocab = self.vectorizer.vocabulary_
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        
        for q in query:
            if q not in vocab:
                continue
            idx = vocab[q]
            
            # Get term frequencies across all docs for this query term
            q_tf = self.X_csc[:, idx].toarray().flatten()
            
            # Optimization: only compute for documents that actually contain the term
            non_zero = q_tf > 0
            if not np.any(non_zero):
                continue
                
            q_tf_nz = q_tf[non_zero]
            doc_len_nz = self.doc_len[non_zero]
            
            num = q_tf_nz * (self.k1 + 1)
            den = q_tf_nz + self.k1 * (1 - self.b + self.b * doc_len_nz / self.avgdl)
            
            scores[non_zero] += self.idf[idx] * (num / den)
            
        return scores
