# cache_manager.py
import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# cache_manager.py - Sentence-level caching

# cache_manager.py - LMDB-based cache (requires: pip install lmdb)

import lmdb
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
import lmdb

class EmbeddingCache:
    """LMDB-based cache - very fast, memory-mapped"""
    
    def __init__(self, cache_dir: str = "./embedding_cache", map_size=100*1024**3):
        """
        Args:
            cache_dir: Cache directory
            map_size: Maximum database size (default 100GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.map_size = map_size
        self.envs = {}  # One environment per model/layer/pooling
    
    def _get_env(self, model_name: str, layer_idx: int, pooling: str):
        """Get or create LMDB environment"""
        safe_name = model_name.replace("/", "_")
        db_name = f"{safe_name}_L{layer_idx}_{pooling}"
        
        if db_name not in self.envs:
            db_path = self.cache_dir / db_name
            db_path.mkdir(exist_ok=True)
            
            env = lmdb.open(
                str(db_path),
                map_size=self.map_size,
                max_dbs=1,
                writemap=True,
                map_async=True,
                lock=False  # Faster for single-process
            )
            self.envs[db_name] = env
        
        return self.envs[db_name]
    
    def _get_key(self, sentence: str) -> bytes:
        """Generate key for a sentence"""
        import hashlib
        normalized = " ".join(sentence.split())
        return hashlib.md5(normalized.encode()).digest()
    
    def get_sentence(self, model_name: str, layer_idx: int, 
                    pooling: str, sentence: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        env = self._get_env(model_name, layer_idx, pooling)
        key = self._get_key(sentence)
        
        with env.begin() as txn:
            value = txn.get(key)
            if value:
                return pickle.loads(value)
        
        return None
    
    def set_sentence(self, model_name: str, layer_idx: int, 
                    pooling: str, sentence: str, embedding: np.ndarray):
        """Cache embedding"""
        env = self._get_env(model_name, layer_idx, pooling)
        key = self._get_key(sentence)
        value = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
        
        with env.begin(write=True) as txn:
            txn.put(key, value)
    
    def close(self):
        """Close all environments"""
        for env in self.envs.values():
            env.close()
        self.envs.clear()




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
