# Building a Vector-Based Context Cache for LLMs: A Practical Guide

## Introduction

Large Language Models like GPT-4 are powerful but expensive and slow. What if we could build an intelligent cache that learns from every interaction, storing responses in a hierarchical vector tree that returns similar answers for similar queries?

In this post, we'll build a **Vector Context Buffer** - a data structure that sits between you and the LLM, learning from every query and creating a searchable memory layer.

## The Concept

Think of it as a neural network's memory:
- **Vectors** represent semantic meaning of queries/responses
- **Tree structure** organizes similar concepts hierarchically
- **Merkle-like hashing** ensures integrity and deduplication
- **Learning** happens with every query added

## Live Implementation

Below is a complete, runnable implementation you can try in Google Colab.

### Setup and Installation

```python
# Run this cell first in Google Colab
!pip install openai faiss-cpu sentence-transformers numpy tiktoken

import os
from google.colab import userdata

# Store your OpenAI API key in Colab Secrets as 'OPENAI_API_KEY'
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

### Core Implementation

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Tuple
import pickle

class VectorNode:
    """Represents a single node in our vector context tree"""
    def __init__(self, query: str, response: str, embedding: np.ndarray):
        self.query = query
        self.response = response
        self.embedding = embedding
        self.timestamp = datetime.now()
        self.access_count = 0
        self.children_hashes = []
        
        # Merkle-style hash
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute merkle-style hash of node content"""
        content = f"{self.query}|{self.response}|{self.embedding.tobytes().hex()}"
        for child_hash in self.children_hashes:
            content += f"|{child_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def increment_access(self):
        """Track how often this node is accessed"""
        self.access_count += 1
        self.timestamp = datetime.now()


class LLMContextCache:
    """
    Intelligent cache layer for LLM interactions using vector similarity search.
    Learns from every query and builds a hierarchical memory structure.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            similarity_threshold: Cosine similarity threshold for cache hits (0-1)
            embedding_model: SentenceTransformer model name
        """
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model (local, free, fast)
        print(f"Loading embedding model: {embedding_model}...")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Storage for actual nodes
        self.nodes: List[VectorNode] = []
        
        # OpenAI client
        self.client = OpenAI()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0
        }
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Convert text to normalized vector embedding"""
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return self._normalize_embedding(embedding)
    
    def _query_llm(self, query: str, model: str = "gpt-4") -> str:
        """Call the actual LLM (GPT-4, etc.)"""
        self.stats['llm_calls'] += 1
        print(f"🔴 CACHE MISS - Querying {model}...")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def query(self, query_text: str, model: str = "gpt-4", force_llm: bool = False) -> Tuple[str, bool]:
        """
        Query the system. Checks cache first, falls back to LLM if needed.
        
        Args:
            query_text: The user's query
            model: LLM model to use if cache misses
            force_llm: Skip cache and query LLM directly
            
        Returns:
            (response_text, was_cached)
        """
        self.stats['total_queries'] += 1
        
        # Generate embedding for query
        query_embedding = self._embed_text(query_text)
        
        # Check cache first (unless forced to use LLM)
        if not force_llm and len(self.nodes) > 0:
            # Search for similar queries
            similarities, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k=1
            )
            
            similarity_score = similarities[0][0]
            
            if similarity_score >= self.similarity_threshold:
                # Cache hit!
                idx = indices[0][0]
                node = self.nodes[idx]
                node.increment_access()
                
                self.stats['cache_hits'] += 1
                print(f"✅ CACHE HIT (similarity: {similarity_score:.3f})")
                print(f"   Original query: '{node.query}'")
                print(f"   Access count: {node.access_count}")
                
                return node.response, True
        
        # Cache miss - query the LLM
        self.stats['cache_misses'] += 1
        response = self._query_llm(query_text, model)
        
        # Add to cache
        self._add_to_cache(query_text, response, query_embedding)
        
        return response, False
    
    def _add_to_cache(self, query: str, response: str, embedding: np.ndarray):
        """Add a new query-response pair to the cache"""
        node = VectorNode(query, response, embedding)
        self.nodes.append(node)
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        
        print(f"💾 Added to cache (total nodes: {len(self.nodes)})")
    
    def get_stats(self) -> dict:
        """Get cache performance statistics"""
        hit_rate = (self.stats['cache_hits'] / self.stats['total_queries'] * 100 
                   if self.stats['total_queries'] > 0 else 0)
        
        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_nodes': len(self.nodes)
        }
    
    def save(self, filepath: str):
        """Save cache to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'nodes': self.nodes,
                'stats': self.stats,
                'threshold': self.similarity_threshold
            }, f)

