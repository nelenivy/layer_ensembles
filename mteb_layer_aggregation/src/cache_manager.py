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
import h5py
logger = logging.getLogger(__name__)


import io
import struct

class LayerEmbeddingStore:
    """
    In-memory pre-computed embedding store with HDF5 persistence.

    PRIMARY ROLE: pre-compute ALL layer embeddings for a known dataset
    in a SINGLE forward pass per batch (via LayerEncoder with
    output_hidden_states=True), keep in RAM, persist to HDF5.

    UNIFIED CACHE INTERFACE: same get_sentence / set_sentence / close
    as EmbeddingCache (SQLite) and LMDBEmbeddingCache, so make_embedding_cache
    can return it transparently.

    EXTRA METHOD: precompute_or_load(texts, dataset_name)
    Must be called before first get_sentence to populate the store.
    Until then, get_sentence returns None (cold cache — LayerEncoder
    will compute and call set_sentence to fill overflow).

    OVERFLOW: sentences that arrive via set_sentence but were not part
    of the original precompute (e.g. queries in retrieval tasks arriving
    separately) are stored in a plain dict side-store.
    """

    def __init__(
        self,
        model_name: str,
        n_layers:   int,
        pooling:    str = "mean",
        batch_size: int = 32,
        device:     str = "cuda",
        cache_dir:  str = ".embeddingcache",
    ):
        self.model_name = model_name
        self.n_layers   = n_layers
        self.pooling    = pooling
        self.batch_size = batch_size
        self.device     = device
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Pre-computed bulk store: layer_idx → np.ndarray (N, d)
        self._embeddings: Dict[int, np.ndarray] = {}
        # sentence text → row index in _embeddings arrays
        self._text_index: Dict[str, int]         = {}

        # Overflow: sentences not covered by precompute
        # sentence → {layer_idx: np.ndarray}
        self._overflow:   Dict[str, Dict[int, np.ndarray]] = {}

        self._is_precomputed = False

    # ------------------------------------------------------------------ #
    # Unified cache interface                                              #
    # ------------------------------------------------------------------ #

    def get_sentence(
        self, model_name: str, layer_idx: int, pooling: str, sentence: str
    ) -> Optional[np.ndarray]:
        # Fast path: bulk precomputed store
        row = self._text_index.get(sentence)
        if row is not None:
            return self._embeddings[layer_idx][row]
        # Overflow path: individually cached sentences
        return self._overflow.get(sentence, {}).get(layer_idx)

    def set_sentence(
        self,
        model_name: str, layer_idx: int, pooling: str,
        sentence:   str, embedding:  np.ndarray,
    ) -> None:
        # Sentences already in the precomputed store need no action
        if sentence in self._text_index:
            return
        # Everything else goes into overflow
        if sentence not in self._overflow:
            self._overflow[sentence] = {}
        self._overflow[sentence][layer_idx] = embedding

    def close(self) -> None:
        pass   # HDF5 written atomically during precompute_or_load; nothing to flush

    # ------------------------------------------------------------------ #
    # Additional: pre-computation                                          #
    # ------------------------------------------------------------------ #
    # In LayerEmbeddingStore:

    def precompute_or_load_from_task(
        self,
        source,                                  # ValidationSplitResolver OR mteb Task
        val_name:        Optional[str] = None,
        max_corpus_size: Optional[int] = None,
    ) -> None:
        """
        Extract texts from a resolver or MTEB task and precompute embeddings.
        See extract_texts_from_task() for full docs.
        """
        from src.utils import extract_texts_from_task

        # Determine dataset_name for HDF5 cache filename
        if hasattr(source, "task_name"):
            dataset_name = source.task_name           # ValidationSplitResolver
        else:
            dataset_name = getattr(source.metadata, "name", str(source))

        texts = extract_texts_from_task(
            source,
            val_name=val_name,
            max_corpus_size=max_corpus_size,
        )

        if not texts:
            logger.warning(
                f"LayerEmbeddingStore: no texts extracted for {dataset_name!r}. "
                f"All embeddings will be computed on-the-fly."
            )
            return

        self.precompute_or_load(texts, dataset_name=dataset_name, split_name=val_name)


    def precompute_or_load(self, texts: List[str], dataset_name: str, split_name: str) -> None:
        """
        Pre-compute or load all layer embeddings for `texts`.

        - If a valid HDF5 file exists for this (model, dataset, pooling):
          loads it into RAM instantly — no model inference.
        - Otherwise: runs LayerEncoder once per batch (single forward pass
          extracts ALL layers via output_hidden_states=True), saves to HDF5.

        Args:
            texts:        All sentences for this dataset split.
            dataset_name: Used to build a stable HDF5 cache filename.
        """
        cache_file = self._cache_file_path(dataset_name, split_name)

        if cache_file.exists() and self._hdf5_is_valid(str(cache_file)):
            logger.info(f"LayerEmbeddingStore: loading from {cache_file}")
            self._load_from_hdf5(str(cache_file), texts)
        else:
            if cache_file.exists():
                logger.warning("LayerEmbeddingStore: corrupt/incomplete HDF5, recomputing")
                cache_file.unlink()
            logger.info(
                f"LayerEmbeddingStore: pre-computing "
                f"{self.n_layers} layers × {len(texts)} texts"
            )
            self._compute_and_save(texts, str(cache_file))

        self._is_precomputed = True

    def get_aggregated(self, texts: List[str], weights: np.ndarray) -> np.ndarray:
        """
        Weighted sum of pre-computed layer embeddings.
        Pure numpy — no model inference. O(n_layers × N × d).
        Only valid after precompute_or_load().
        """
        if not self._is_precomputed:
            raise RuntimeError("Call precompute_or_load() first")
        indices = [self._text_index[t] for t in texts]
        return sum(
            weights[i] * self._embeddings[i][indices]
            for i in range(self.n_layers)
        )

    def is_ready(self) -> bool:
        return self._is_precomputed

    # ------------------------------------------------------------------ #
    # Internal: compute via LayerEncoder (single forward pass per batch)  #
    # ------------------------------------------------------------------ #

    def _compute_and_save(self, texts: List[str], cache_file: str) -> None:
        from src.aggregated_encoder import LayerEncoder   # local import avoids circular

        self._text_index = {t: i for i, t in enumerate(texts)}
        accum: Dict[int, List[np.ndarray]] = {i: [] for i in range(self.n_layers)}

        # No cache here — we ARE the cache; no point writing to a second store
        layer_enc = LayerEncoder(
            model_name=self.model_name,
            pooling=self.pooling,
            batch_size=self.batch_size,
            device=self.device,
            use_cache=False,
        )

        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for b, start in enumerate(range(0, len(texts), self.batch_size)):
            batch = texts[start : start + self.batch_size]
            logger.info(f"  batch {b + 1}/{n_batches} ({len(batch)} texts)")

            # Single forward pass → list of n_layers arrays, each (batch, d)
            all_layer_embs = layer_enc.encode_batch(batch, return_all_layers=True)

            for layer_idx, emb in enumerate(all_layer_embs):
                accum[layer_idx].append(emb)

        layer_embs = {
            i: np.concatenate(accum[i], axis=0)
            for i in range(self.n_layers)
        }

        # Atomic write: tmp → rename
        tmp = cache_file + ".tmp"
        self._save_to_hdf5(tmp, texts, layer_embs)
        os.replace(tmp, cache_file)

        self._embeddings = layer_embs
        logger.info(f"LayerEmbeddingStore: saved → {cache_file}")

    # ------------------------------------------------------------------ #
    # Internal: HDF5                                                       #
    # ------------------------------------------------------------------ #

    def _save_to_hdf5(
        self, path: str, texts: List[str], layer_embs: Dict[int, np.ndarray]
    ) -> None:
        with h5py.File(path, "w") as f:
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset("texts", data=[t.encode("utf-8") for t in texts], dtype=dt)
            grp = f.create_group("layers")
            for layer_idx, embs in layer_embs.items():
                grp.create_dataset(
                str(layer_idx),
                data=embs,
                dtype=embs.dtype,               # ← explicit: float32 or float64, whatever came out of the model
                compression="lzf",
                chunks=(min(256, len(texts)), embs.shape[1]),
            )
            # Written LAST — missing sentinel = file is corrupt/incomplete
            f.create_dataset("complete", data=np.array([1], dtype=np.int8))

    def _load_from_hdf5(self, path: str, texts: List[str]) -> None:
        with h5py.File(path, "r") as f:
            stored_texts = [t.decode("utf-8") for t in f["texts"][:]]
            self._text_index = {t: i for i, t in enumerate(stored_texts)}
            self._embeddings = {
                int(k): f["layers"][k][:]   # dtype preserved exactly as saved
                for k in f["layers"].keys()
            }
        missing = [t for t in texts if t not in self._text_index]
        if missing:
            raise KeyError(
                f"{len(missing)} texts not found in HDF5 store — "
                f"delete {path} to trigger recomputation"
            )

    @staticmethod
    def _hdf5_is_valid(path: str) -> bool:
        try:
            with h5py.File(path, "r") as f:
                return (
                    "complete"   in f and int(f["complete"][0]) == 1
                    and "keys"   not in f           # guard against old InMemoryEmbeddingCache format
                    and "layers" in f
                    and "texts"  in f
                )
        except Exception:
            return False

    def _cache_file_path(self, dataset_name: str, split_name: str) -> Path:
        key = f"{self.model_name}_{dataset_name}_{split_name}_{self.pooling}_{self.n_layers}"
        h   = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"layer_store_{h}.h5"


class LMDBEmbeddingCache:
    """
    Fast drop-in replacement for SQLite embedding cache.

    - Memory-mapped reads: ~10-20x faster than SQLite for embedding lookups.
    - Corruption safety: any lmdb.Error during read or open triggers a full
      cache wipe and fresh start — no silent bad data, no crashes.
    - Auto-growing map: doubles map_size on MapFullError instead of crashing.
    - Same get/set interface as existing SQLite cache.
    """

    _NUMPY_MAGIC = b"NP"   # 2-byte magic prefix to detect valid entries

    def __init__(
        self,
        cache_dir: str,
        map_size: int = 10 * 1024 ** 3,    # 10 GB virtual address space
        readonly: bool = False,
    ):
        self.cache_path = cache_dir
        self.map_size   = map_size
        self.readonly   = readonly
        self._env       = None
        self._open()

    # ------------------------------------------------------------------ #
    # Public interface (same as SQLite cache)                              #
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[np.ndarray]:
        try:
            with self._env.begin() as txn:
                data = txn.get(key.encode())
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.warning(f"LMDB read error (key={key!r}): {e} — invalidating cache")
            self._invalidate_and_reopen()
            return None

    def set(self, key: str, embedding: np.ndarray) -> None:
        data = self._serialize(embedding)
        self._write(key.encode(), data)

    def __contains__(self, key: str) -> bool:
        try:
            with self._env.begin() as txn:
                return txn.get(key.encode()) is not None
        except Exception:
            return False

    def close(self) -> None:
        if self._env:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        import lmdb
        try:
            self._env = lmdb.open(
                self.cache_path,
                map_size=self.map_size,
                max_readers=128,
                readonly=self.readonly,
                lock=not self.readonly,
                readahead=False,    # disable for random-access workloads
                meminit=False,      # skip zero-fill for speed
                metasync=False,     # don't fsync meta on every commit
                sync=False,         # async flushes (safe enough; we detect corruption)
            )
        except Exception as e:
            logger.warning(f"LMDB open failed at {self.cache_path!r}: {e} — recreating")
            shutil.rmtree(self.cache_path, ignore_errors=True)
            import lmdb as _lmdb
            self._env = _lmdb.open(self.cache_path, map_size=self.map_size,
                                   max_readers=128, readahead=False, meminit=False)

    def _write(self, key: bytes, data: bytes) -> None:
        import lmdb
        try:
            with self._env.begin(write=True) as txn:
                txn.put(key, data)
        except lmdb.MapFullError:
            # Grow map size and retry
            self._env.close()
            self.map_size *= 2
            logger.info(f"LMDB map full — growing to {self.map_size // 1024**3} GB")
            self._open()
            with self._env.begin(write=True) as txn:
                txn.put(key, data)
        except Exception as e:
            logger.warning(f"LMDB write error: {e} — invalidating cache")
            self._invalidate_and_reopen()

    def _invalidate_and_reopen(self) -> None:
        self.close()
        shutil.rmtree(self.cache_path, ignore_errors=True)
        logger.warning(f"Recreated LMDB cache at {self.cache_path!r}")
        self._open()

    def _serialize(self, arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        buf.write(self._NUMPY_MAGIC)
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    def _deserialize(self, data: bytes) -> np.ndarray:
        if data[:2] != self._NUMPY_MAGIC:
            raise ValueError("Invalid LMDB cache entry: missing magic bytes")
        return np.load(io.BytesIO(data[2:]), allow_pickle=False)

# cache_manager.py - SQLite version (more robust)


class SQLiteEmbeddingCache:
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
