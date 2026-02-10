# cache_manager.py
import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# cache_manager.py - Sentence-level caching

# cache_manager.py - Robust LMDB implementation

import lmdb
import numpy as np
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import logging
import sqlite3

logger = logging.getLogger(__name__)

# cache_manager.py - SQLite version (more robust)


class EmbeddingCache:
    """SQLite-based cache - more robust than LMDB"""
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.connections = {}
        self.memory_cache = {}
        self.max_memory_items = 10000
    
    def _get_db_path(self, model_name: str, layer_idx: int, pooling: str) -> Path:
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}_L{layer_idx}_{pooling}.db"
    
    def _get_connection(self, model_name: str, layer_idx: int, pooling: str):
        db_path = self._get_db_path(model_name, layer_idx, pooling)
        key = str(db_path)
        
        if key not in self.connections:
            try:
                conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
                conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
                conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        sentence_hash TEXT PRIMARY KEY,
                        embedding BLOB
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_hash ON embeddings(sentence_hash)')
                conn.commit()
                self.connections[key] = conn
            except sqlite3.Error as e:
                logger.error(f"SQLite error: {e}")
                # Delete corrupted database
                if db_path.exists():
                    db_path.unlink()
                    return self._get_connection(model_name, layer_idx, pooling)
                raise
        
        return self.connections[key]
    
    def _get_key(self, sentence: str) -> str:
        normalized = " ".join(sentence.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_sentence(self, model_name: str, layer_idx: int, 
                    pooling: str, sentence: str) -> Optional[np.ndarray]:
        mem_key = f"{model_name}_{layer_idx}_{pooling}_{sentence[:50]}"
        if mem_key in self.memory_cache:
            return self.memory_cache[mem_key]
        
        try:
            conn = self._get_connection(model_name, layer_idx, pooling)
            key = self._get_key(sentence)
            
            cursor = conn.execute('SELECT embedding FROM embeddings WHERE sentence_hash = ?', (key,))
            row = cursor.fetchone()
            
            if row:
                embedding = pickle.loads(row[0])
                
                if len(self.memory_cache) >= self.max_memory_items:
                    self.memory_cache.pop(next(iter(self.memory_cache)))
                self.memory_cache[mem_key] = embedding
                
                return embedding
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set_sentence(self, model_name: str, layer_idx: int, 
                    pooling: str, sentence: str, embedding: np.ndarray):
        mem_key = f"{model_name}_{layer_idx}_{pooling}_{sentence[:50]}"
        self.memory_cache[mem_key] = embedding
        
        try:
            conn = self._get_connection(model_name, layer_idx, pooling)
            key = self._get_key(sentence)
            blob = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
            
            conn.execute('''
                INSERT OR REPLACE INTO embeddings (sentence_hash, embedding)
                VALUES (?, ?)
            ''', (key, blob))
            conn.commit()
        
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def close(self):
        for conn in self.connections.values():
            try:
                conn.close()
            except:
                pass
        self.connections.clear()
        self.memory_cache.clear()




class QualityCache:
    """Cache for layer quality scores by (model, pooling, dataset)"""
    
    def __init__(self, cache_dir: str = "./quality_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, model_name: str, pooling: str, task_name: str) -> str:
        """Generate cache key for quality scores"""
        key_str = f"{model_name}_{pooling}_{task_name}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"quality_{key_hash}.pkl"
    
    def get(self, model_name: str, pooling: str, task_name: str) -> Optional[np.ndarray]:
        """Retrieve cached quality scores"""
        cache_file = self.cache_dir / self._get_cache_key(model_name, pooling, task_name)
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                print(f"✓ Quality cache hit: {task_name}, {pooling}")
                return data['layer_quality']
        return None
    
    def set(self, model_name: str, pooling: str, task_name: str, 
            layer_quality: np.ndarray):
        """Store quality scores in cache"""
        cache_file = self.cache_dir / self._get_cache_key(model_name, pooling, task_name)
        
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'layer_quality': layer_quality,
                'model_name': model_name,
                'pooling': pooling,
                'task_name': task_name
            }, f)
        print(f"✓ Quality cached: {task_name}, {pooling}")
