from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from mteb.models import EncoderProtocol
import mteb
import logging  
import warnings
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
import random
from collections import defaultdict
from src.cache_manager import QualityCache, EmbeddingCache
from src.metrics.calculator import SimilarityCalculator, AVAILABLE_METRICS
from collections.abc import Iterable
from datasets.features.features import Value  # HuggingFace column types
from src.aggregated_encoder import AggregatedEncoder 
from types import SimpleNamespace
import json
from pathlib import Path  
from src.fisher_constrained_optimization import fisher_gradient_descent_trust_region_efficient
from src.validation_split_resolver import ValidationSplitResolver

def split_retrieval_data(queries, corpus, qrels, val_size=0.2, seed=42, qrels_key='qrels'):
    """
    Split retrieval data into train/test sets.
    
    Parameters:
    - queries: dict {query_id: query_text}
    - corpus: dict {doc_id: doc_text}
    - qrels: dict {query_id: {doc_id: relevance_score}}
    - val_size: fraction for validation (0.0 to 1.0)
    - seed: random seed
    
    Returns:
    - dict with 'train' and 'test' splits
    """
    random.seed(seed)
    
    # Get all query IDs and shuffle
    all_query_ids = list(qrels.keys())
    random.shuffle(all_query_ids)
    
    # Split query IDs
    split_point = int(len(all_query_ids) * (1 - val_size))
    train_qids = set(all_query_ids[:split_point])
    val_qids = set(all_query_ids[split_point:])
    
    # Split queries
    train_queries = {qid: queries[qid] for qid in train_qids if qid in queries}
    val_queries = {qid: queries[qid] for qid in val_qids if qid in queries}
    
    # Split qrels
    train_qrels = {qid: qrels[qid] for qid in train_qids if qid in qrels}
    val_qrels = {qid: qrels[qid] for qid in val_qids if qid in qrels}
    
    # Corpus is shared (no split needed)
    return {
        'train': {
            'queries': train_queries,
            'corpus': corpus,
            qrels_key: train_qrels
        },
        'test': {
            'queries': val_queries,
            'corpus': corpus,
            qrels_key: val_qrels
        }
    }

class SingleLayerEncoder(EncoderProtocol):
    """Encoder with sentence-level caching"""
    
    def __init__(self, model, tokenizer, layer_idx, pooling, device, 
                 batch_size, model_name, use_cache=True, cache_dir="./embedding_cache"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = model.config.hidden_size
        self.model_name = model_name
        
        if use_cache:
            self.embedding_cache = EmbeddingCache(cache_dir)
        else:
            self.embedding_cache = None

        if model_name:
            custom_name = f"{model_name.replace('/', '_')}_layer{layer_idx}_{pooling}"
        else:
            custom_name = f"SingleLayer{layer_idx}_{pooling}"
        
        self.mteb_model_meta = SimpleNamespace(
            name=custom_name,
            revision="main",
            release_date=None,
            languages=None
        )
    
    def _encode_batch(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Encode batch with sentence-level caching"""
        if not sentences:
            return None
        
        # Normalize sentences
        print('len sentences:', len(sentences))
        sentences = [str(s).strip() for s in sentences if s is not None]
        sentences = [s for s in sentences if s]
        
        if not sentences:
            return None
        
        # Check cache for each sentence
        results = [None] * len(sentences)
        uncached_indices = []
        uncached_sentences = []
        
        if self.embedding_cache:
            for i, sentence in enumerate(sentences):
                cached = self.embedding_cache.get_sentence(
                    self.model_name, self.layer_idx, self.pooling, sentence
                )
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_sentences.append(sentence)
        else:
            uncached_indices = list(range(len(sentences)))
            uncached_sentences = sentences
        
        # Compute uncached embeddings
        if uncached_sentences:
            try:
                inputs = self.tokenizer(
                    uncached_sentences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.hidden_states[self.layer_idx]
                    
                    if self.pooling == "cls":
                        pooled = hidden_states[:, 0, :]
                    else:  # mean
                        mask = inputs['attention_mask'].unsqueeze(-1).float()
                        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    
                    new_embeddings = pooled.cpu().numpy()
                
                # Cache and store results
                for i, emb in enumerate(new_embeddings):
                    original_idx = uncached_indices[i]
                    results[original_idx] = emb
                    
                    if self.embedding_cache:
                        self.embedding_cache.set_sentence(
                            self.model_name, self.layer_idx, self.pooling,
                            sentences[original_idx], emb
                        )
            
            except Exception as e:
                print(f"Error encoding batch: {e}")
                return None
        
        return np.vstack(results)
    
    def _extract_sentences_from_batch(self, batch):
        """Extract sentences from DataLoader batch."""
        if isinstance(batch, dict):
            for key in ["text", "sentence", "sentences", "query", "passage", "title", "content"]:
                if key in batch and batch[key] is not None:
                    return batch[key]
            for v in batch.values():
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                    return v
        elif isinstance(batch, (list, tuple)):
            return batch
        else:
            return [str(batch)]
        return []
    
    def encode(
        self,
        sentences: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences using single layer (MTEB interface)."""
        # Handle DataLoader (MTEB 2.0 can pass this)
        if isinstance(sentences, DataLoader):
            print("isinstance(sentences, DataLoader)")
            all_embeddings = []
            for batch in sentences:
                batch_sentences = self._extract_sentences_from_batch(batch)
                if not batch_sentences:
                    continue
                batch_embs = self._encode_batch(batch_sentences)
                if batch_embs is not None and len(batch_embs) > 0:
                    all_embeddings.append(batch_embs)
            if not all_embeddings:
                return np.zeros((0, self.hidden_size), dtype=np.float32)
            return np.vstack(all_embeddings)
        
        # Handle string input
        if isinstance(sentences, str):
            print("isinstance(sentences, str):")
            sentences = [sentences]
        
        # Handle list of sentences
        if not sentences or len(sentences) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        print('batch size', batch_size)
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embs = self._encode_batch(batch)
            if batch_embs is not None:
                all_embeddings.append(batch_embs)
        
        if not all_embeddings:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        return np.vstack(all_embeddings)
    
    def encode_queries(
        self,
        queries: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode queries (MTEB interface)."""
        return self.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]], DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode corpus (MTEB interface)."""
        # Handle DataLoader
        if isinstance(corpus, DataLoader):
            return self.encode(corpus, batch_size=batch_size, **kwargs)
        
        # Handle dict format (common in MTEB corpus)
        if isinstance(corpus, list) and corpus and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self.encode(sentences, batch_size=batch_size, **kwargs)
        
        return self.encode(corpus, batch_size=batch_size, **kwargs)
    
    def similarity(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between queries and corpus."""
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        return queries_norm @ corpus_norm.T
    
    def similarity_pairwise(self, sentences1: np.ndarray, sentences2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity."""
        s1_norm = sentences1 / (np.linalg.norm(sentences1, axis=1, keepdims=True) + 1e-8)
        s2_norm = sentences2 / (np.linalg.norm(sentences2, axis=1, keepdims=True) + 1e-8)
        return np.sum(s1_norm * s2_norm, axis=1)


def try_single_sentence_keys(data_dict):
    print(5)
    for field in ['text', 'sentence', 'query', 'passage', 'title']:
        #print(list(data_dict.keys()), field)
        print(type(data_dict))
        is_in = (field in data_dict) if isinstance(data_dict, dict) else (field in data_dict.column_names)
        if is_in:
            sentences = data_dict[field]
            print("Try single sentence fields first", field, len(sentences))
            # Check for any iterable (but not string or dict)
            if isinstance(sentences, Iterable) and not isinstance(sentences, (str, dict, bytes)):
                return list(sentences)
    
    return None

        
def calculate_similarity_matrix_from_task(
    model,
    tokenizer,
    model_name: str,
    task_name: str,
    dataset_to_use: Dict,
    val_name: str,
    n_layers: int,
    similarity_metric: str,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    similarity_sample_size: int = 3000,
    similarity_num_trials: int = 5,
    similarity_cache_dir: str = "./similarity_cache",
    use_emb_cache: bool = True,
    emb_cache_dir: str = "./embs_cache",
    verbose: int = 1
) -> np.ndarray:
    """
    Calculate similarity matrix using already-loaded task and validation split.
    
    Args:
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        model_name: Model name for caching
        task_name: Task name for caching
        dataset_to_use: Already processed dataset dict
        val_name: Name of validation split to use
        n_layers: Number of layers in the model
        similarity_metric: Metric to use (CKA, RSA, JaccardSimilarity, DistanceCorrelation)
        pooling: Pooling strategy
        batch_size: Batch size for encoding
        device: Device for computation
        similarity_sample_size: Samples per Monte Carlo trial
        similarity_num_trials: Number of Monte Carlo trials
        similarity_cache_dir: Cache directory for similarity matrices
        use_emb_cache: Whether to use embedding cache
        emb_cache_dir: Embedding cache directory
        verbose: Verbosity level
        
    Returns:
        similarity_matrix: Array of shape (n_layers, n_layers)
    """
    
    # Check cache first
    os.makedirs(similarity_cache_dir, exist_ok=True)
    model_name_safe = model_name.replace("/", "_")
    task_name_safe = task_name.replace("/", "_")
    sim_cache_path = os.path.join(
        similarity_cache_dir,
        f"{model_name_safe}_{task_name_safe}_{similarity_metric}_{pooling}.pkl"
    )
    
    if os.path.exists(sim_cache_path):
        if verbose >= 1:
            print(f"\nLoading cached similarity matrix from {sim_cache_path}")
        with open(sim_cache_path, 'rb') as f:
            sim_data = pickle.load(f)
        return sim_data['mean']
    
    if verbose >= 1:
        print(f"\nCalculating {similarity_metric} similarity matrix for {task_name}...")
    
    # Extract sentences from the validation split
    val_data = dataset_to_use[val_name]
    sentences = []

    if hasattr(val_data, 'to_dict'):
        data_dict = val_data.to_dict()
    elif isinstance(val_data, dict):
        data_dict = val_data
    else:
        data_dict = {'text': val_data}

    print("data_dict", list(data_dict.keys()))
    # Try single sentence fields first
    sentences = try_single_sentence_keys(data_dict)

    # If no sentences found, try sentence pair fields (for STS tasks)
    if not sentences:
        if 'sentence1' in data_dict and 'sentence2' in data_dict:
            sentences1 = data_dict['sentence1']
            sentences2 = data_dict['sentence2']
            if isinstance(sentences1, list) and isinstance(sentences2, list):
                if len(sentences1) == 1:
                    sentences1 = sentences1[0]

                if len(sentences2) == 1:
                    sentences2 = sentences2[0]
                sentences = sentences1 + sentences2
                print("If no sentences found, try sentence pair fields (for STS tasks)", len(sentences))
                if verbose >= 1:
                    print(f"Using sentence pairs from STS task (combined {len(sentences1)} + {len(sentences2)} sentences)")

    # If still no sentences, try retrieval task structure (corpus + queries)
    if not sentences:
        if 'corpus' in data_dict and 'queries' in data_dict:
            corpus = data_dict['corpus']
            queries = data_dict['queries']
            print(type(queries))
            # Extract texts from corpus (dict of dicts)
            if isinstance(corpus, dict):# or isinstance(corpus, Dataset):
                print(1)
                print(list(corpus.items())[:10])
                for doc_id, doc_content in corpus.items():
                    if isinstance(doc_content, dict):
                        # Try text field first, then title
                        text = doc_content.get('text', doc_content.get('title', ''))
                        if text:
                            sentences.append(str(text))
                    elif isinstance(doc_content, str):
                        sentences.append(doc_content)
            else:
                print(2)
                sentences = try_single_sentence_keys(corpus)
            
            # Extract texts from queries (dict of strings)
            if isinstance(queries, dict):# or isinstance(queries, Dataset):
                print(3)
                for query_id, query_text in queries.items():
                    if isinstance(query_text, str):
                        sentences.append(query_text)
            else:
                print(4)
                print(queries['text'])
                curr = try_single_sentence_keys(queries)
                if curr:
                    sentences += curr
            
            if sentences and verbose >= 1:
                print(f"Using retrieval task data (corpus + queries): {len(sentences)} sentences total")

    if not sentences:
        raise ValueError(f"Could not extract sentences from validation split '{val_name}'. Available fields: {list(data_dict.keys())}")

    print("sentences", sentences[:10])
    #print(split_data)
    # Limit sentences FIRST
    if len(sentences) > similarity_sample_size * 2:
        sentences = sentences[:similarity_sample_size * 2]
    if verbose >= 1:
        print(f"Limited to {len(sentences)} sentences for similarity calculation")
    if verbose >= 1:
        print(f"Getting embeddings for all {n_layers} layers using MTEB API...")
    
    # Get embeddings for all layers using MTEB's encoding
    all_layer_embeddings = {}
    for layer_idx in range(n_layers):
        # Create single-layer encoder
        encoder = SingleLayerEncoder(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            pooling=pooling,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            use_cache=use_emb_cache,
            cache_dir=emb_cache_dir
        )

        # Encode sentences - will hit embedding cache!
        layer_embs = encoder.encode(sentences)
        all_layer_embeddings[layer_idx] = torch.from_numpy(layer_embs)
        # Extract sentences from the validation split
        # Encode using the encoder - handles all formats automatically
        # if isinstance(split_data, dict) and 'corpus' in split_data:
        #     # Retrieval task - encode corpus
        #     embeddings = encoder.encode_corpus(
        #         split_data['corpus'],
        #         batch_size=batch_size
        #     )
        # else:
        #     # Other tasks - encode directly
        #     embeddings = encoder.encode(
        #         split_data,
        #         batch_size=batch_size
        #     )
        print(len(layer_embs))
        
        all_layer_embeddings[layer_idx] = torch.from_numpy(layer_embs)
    
    
    if verbose >= 1:
        print(f"Calculating {similarity_metric} similarity matrix...")
    
    # Calculate similarity matrix using calculator.py
    calculator = SimilarityCalculator(
        sample_size=similarity_sample_size,
        num_trials=similarity_num_trials,
        seed=42
    )
    
    results = calculator.calculate_all(
        embeddings=all_layer_embeddings,
        metrics=[similarity_metric]
    )
    
    similarity_matrix = results[similarity_metric]['mean']
    
    # Cache the similarity matrix
    cache_data = {
        'mean': similarity_matrix,
        'std': results[similarity_metric]['std'],
        'metric': similarity_metric,
        'task_name': task_name,
        'model_name': model_name,
        'pooling': pooling
    }
    
    with open(sim_cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    if verbose >= 1:
        print(f"Cached similarity matrix to {sim_cache_path}")
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Similarity matrix stats: mean={similarity_matrix.mean():.4f}, std={similarity_matrix.std():.4f}")
    
    return similarity_matrix
        
def compute_dataset_specific_layer_quality(
    model_name: str,
    task_name: str,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    verbose: int = 1,
    use_cache: bool = True, 
    cache_dir: str = "./quality_cache",
    use_emb_cache: bool = True, 
    emb_cache_dir: str = "./embs_cache",
    calc_similarity_matrix: bool = False,
    similarity_metric: Optional[str] = None,
    similarity_sample_size: int = 3000,
    similarity_num_trials: int = 5,
    similarity_cache_dir: str = "./similarity_cache",
    calc_hessian_matrix: bool = False,
    hessian_cache_dir: str = "./hessian_cache",
    hessian_method: str = 'fisher', #"least_squares",  # "least_squares" or "fisher"
    gradient_descent=False,
    num_random_points: int = 0  # For least-squares: additional random evaluations
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute layer quality by evaluating each layer on a specific MTEB task with caching support.
    Optionally calculate similarity matrix or Hessian matrix using the same task/split data.
    
    Args:
        model_name: HuggingFace model name
        task_name: MTEB task name
        pooling: Pooling strategy ('mean' or 'cls')
        batch_size: Batch size for encoding
        device: Device for computation
        verbose: Verbosity level (0, 1, 2)
        use_cache: Use quality cache
        cache_dir: Quality cache directory
        use_emb_cache: Use embedding cache
        emb_cache_dir: Embedding cache directory
        calc_similarity_matrix: Whether to calculate similarity matrix
        similarity_metric: Similarity metric (CKA, RSA, JaccardSimilarity, DistanceCorrelation)
        similarity_sample_size: Samples per Monte Carlo trial
        similarity_num_trials: Number of Monte Carlo trials
        similarity_cache_dir: Similarity cache directory
        calc_hessian_matrix: Whether to calculate Hessian matrix
        hessian_cache_dir: Hessian cache directory
        hessian_method: Method for Hessian ("least_squares" or "fisher")
        num_random_points: Additional random evaluation points for least-squares
        
    Returns:
        layer_qualities: Array of shape (n_layers,) with quality scores
        OR
        (layer_qualities, similarity_matrix): Tuple if calc_similarity_matrix=True
        OR
        (linear_coeff, -hessian_matrix): Tuple if calc_hessian_matrix=True
            Returns linear coefficients and negative Hessian from quadratic fit
    """

    # Check cache first for quality
    cached_quality = None
    need_task_for_computation = calc_similarity_matrix or calc_hessian_matrix or gradient_descent
    
    if use_cache:
        quality_cache = QualityCache(cache_dir)
        cached_quality = quality_cache.get(model_name, pooling, task_name)
        
        if cached_quality is not None:
            # Check similarity cache if needed
            if calc_similarity_matrix and similarity_metric:
                os.makedirs(similarity_cache_dir, exist_ok=True)
                model_name_safe = model_name.replace("/", "_")
                task_name_safe = task_name.replace("/", "_")
                sim_cache_path = os.path.join(
                    similarity_cache_dir,
                    f"{model_name_safe}_{task_name_safe}_{similarity_metric}_{pooling}.pkl"
                )
                if os.path.exists(sim_cache_path):
                    if verbose >= 1:
                        print(f"Loading cached similarity matrix from {sim_cache_path}")
                    with open(sim_cache_path, 'rb') as f:
                        sim_data = pickle.load(f)
                    return cached_quality, sim_data['mean']
                # Quality cached but similarity not - will compute similarity below
                
            # Check Hessian cache if needed
            elif calc_hessian_matrix:
                os.makedirs(hessian_cache_dir, exist_ok=True)
                model_name_safe = model_name.replace("/", "_")
                task_name_safe = task_name.replace("/", "_")
                
                # Cache filename depends on method
                if hessian_method == "fisher":
                    hessian_cache_path = os.path.join(
                        hessian_cache_dir,
                        f"{model_name_safe}_{task_name_safe}_hessian_fisher_persample_{pooling}.pkl"
                    )
                else:  # least_squares
                    hessian_cache_path = os.path.join(
                        hessian_cache_dir,
                        f"{model_name_safe}_{task_name_safe}_hessian_lsq_{pooling}.pkl"
                    )
                
                if os.path.exists(hessian_cache_path):
                    if verbose >= 1:
                        print(f"Loading cached Hessian from {hessian_cache_path}")
                    with open(hessian_cache_path, 'rb') as f:
                        hessian_data = pickle.load(f)
                    # Return linear coefficients and negative Hessian
                    center_weights = (cached_quality == np.max(cached_quality)).astype(float)
                    return center_weights, hessian_data['coeff'], -hessian_data['hessian']
                
                # ✅ FIX: Hessian not cached - DON'T return early!
                # Fall through to load task and compute Hessian
            
            # Only return early if we don't need to compute anything else
            elif not need_task_for_computation:
                return cached_quality

    if verbose >= 1:
        print(f"\n{'='*70}")
        if cached_quality is not None:
            print(f"Layer quality cached. Loading task for {task_name} to compute Hessian/similarity...")
        else:
            print(f"Computing layer quality on: {task_name}")
        print(f"{'='*70}\n")
    
    # Load task
    resolver = ValidationSplitResolver(task_name, verbose=verbose)

    # Access any property; loading happens only on the first call.
    val_name     = resolver.val_name     # triggers load once
    dataset      = resolver.dataset
    task_type    = resolver.task_type
    
    # Load model once for all layers
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    ).to(device).eval()
    
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    layer_qualities = []
    
    # Evaluate each layer (only if not cached)
    if cached_quality is None:
        if verbose >= 1:
            print(f"Evaluating {n_layers} layers...")
        
        with warnings.catch_warnings():
            # Suppress all MTEB-related and HTTP logging
            loggers_to_suppress = [
                'mteb', 
                'datasets', 
                'transformers',
                'httpx',           
                'urllib3',
                'filelock',
                'huggingface_hub'
            ]
            original_levels = {}
            
            for logger_name in loggers_to_suppress:
                logger = logging.getLogger(logger_name)
                original_levels[logger_name] = logger.level
                logger.setLevel(logging.INFO)  # Only critical errors
                
            # Evaluate each layer
            for layer_idx in range(n_layers):
                if verbose >= 1:
                    print(f"  Layer {layer_idx}/{n_layers-1}...", end=" ", flush=True)

                # Create encoder for this layer
                encoder = SingleLayerEncoder(
                    model=model,
                    tokenizer=tokenizer,
                    layer_idx=layer_idx,
                    pooling=pooling,
                    device=device,
                    batch_size=batch_size,
                    model_name=model_name,
                    use_cache=use_emb_cache, 
                    cache_dir=emb_cache_dir
                )

                # Evaluate on MTEB task
                results = mteb.evaluate(
                    model=encoder,
                    tasks=[task],
                    encode_kwargs={"batch_size": batch_size},
                    show_progress_bar=False,
                    overwrite_strategy="always"
                )

                # Extract main score for the requested split
                task_result = results[0]
                scores = task_result.scores[val_name][0]
                quality = scores.get('main_score', 0.0)
                layer_qualities.append(quality)
                
                if verbose >= 1:
                    print(f"{quality:.4f}")
            
            # Restore original logging levels
            for logger_name, level in original_levels.items():
                logging.getLogger(logger_name).setLevel(level)
        
        layer_qualities = np.array(layer_qualities, dtype=np.float32)
        
        if verbose >= 1:
            print(f"\nLayer qualities computed!")
            print(f"Best layer: {np.argmax(layer_qualities)} (score: {np.max(layer_qualities):.4f})")
            print(f"Mean quality: {np.mean(layer_qualities):.4f}")

        # Cache the result
        if use_cache:
            quality_cache.set(model_name, pooling, task_name, layer_qualities)
    else:
        layer_qualities = cached_quality
        if verbose >= 1:
            print(f"\nUsing cached layer qualities")
            print(f"Best layer: {np.argmax(layer_qualities)} (score: {np.max(layer_qualities):.4f})")
            print(f"Mean quality: {np.mean(layer_qualities):.4f}")
        
    # Calculate similarity matrix if requested
    if calc_similarity_matrix and similarity_metric:
        similarity_matrix = calculate_similarity_matrix_from_task(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            task_name=resolver.task_name,
            dataset_to_use=resolver.dataset,
            val_name=resolver.val_name,
            n_layers=n_layers,
            similarity_metric=similarity_metric,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            similarity_sample_size=similarity_sample_size,
            similarity_num_trials=similarity_num_trials,
            similarity_cache_dir=similarity_cache_dir,
            use_emb_cache=use_emb_cache,
            emb_cache_dir=emb_cache_dir,
            verbose=verbose
        )
        return layer_qualities, similarity_matrix
    
    # Calculate Hessian matrix if requested
    if gradient_descent:
        weights, quality_hist, weight_hist = fisher_gradient_descent_trust_region_efficient(
            model_name=model_name,
            dataset_resolver=resolver,
            n_layers=n_layers,
            layer_qualities=layer_qualities,
            max_steps=5,
            verbose=2
        )

        return weights
    if calc_hessian_matrix:
        if hessian_method == "least_squares":
            c, coeff, hessian_matrix = calculate_hessian_matrix_lsq(
                model_name=model_name,
                task=resolver.task,
                task_name=resolver.task_name,
                dataset_to_use=resolver.dataset,
                val_name=resolver.val_name,
                n_layers=n_layers,
                layer_qualities=layer_qualities,
                pooling=pooling,
                batch_size=batch_size,
                device=device,
                hessian_cache_dir=hessian_cache_dir,
                use_emb_cache=use_emb_cache,
                emb_cache_dir=emb_cache_dir,
                verbose=verbose,
                num_random_points=num_random_points
            )
        elif hessian_method == "fisher":
            c, coeff, hessian_matrix = calculate_hessian_matrix_fisher(
                model_name=model_name,
                task=resolver.task,
                task_name=resolver.task_name,
                dataset_to_use=resolver.dataset,
                val_name=resolver.val_name,
                n_layers=n_layers,
                layer_qualities=layer_qualities,
                pooling=pooling,
                batch_size=batch_size,
                device=device,
                hessian_cache_dir=hessian_cache_dir,
                use_emb_cache=use_emb_cache,
                emb_cache_dir=emb_cache_dir,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown hessian_method: {hessian_method}. Use 'least_squares' or 'fisher'")
        
        # Return linear coefficients and negative Hessian
        # (don't return constant c)
        return c, coeff, -hessian_matrix
    
    # Default: just return layer qualities
    return layer_qualities




def calculate_hessian_matrix(
    model_name: str,
    task,
    task_name: str,
    dataset_to_use: Dict,
    val_name: str,
    n_layers: int,
    layer_qualities: np.ndarray,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    hessian_cache_dir: str = "./hessian_cache",
    use_emb_cache: bool = True,
    emb_cache_dir: str = "./embs_cache",
    verbose: int = 1
) -> np.ndarray:
    """
    Calculate Hessian matrix of ensemble error using finite differences.
    Uses the existing AggregatedEncoder class.

    The error function is: err = f(w1, w2, ..., wn)
    where ensemble = sum_i wi * emb_i

    We evaluate at:
    - Individual layers: (1, 0, 0, ..., 0), (0, 1, 0, ..., 0), etc. [already have these]
    - Pairwise combinations: (0.5, 0.5, 0, ..., 0) for all pairs (i, j)

    Then compute Hessian using finite differences:
    H_ij ≈ [f(0.5*e_i + 0.5*e_j) - f(e_i) - f(e_j) + f(0)] / (0.5 * 0.5)

    For diagonal elements:
    H_ii ≈ [f(e_i) - 2*f(0.5*e_i) + f(0)] / (0.5)²

    Args:
        model_name: Model name
        task: MTEB task object
        task_name: Task name for caching
        dataset_to_use: Dataset dict
        val_name: Validation split name
        n_layers: Number of layers
        layer_qualities: Quality scores for individual layers (shape: n_layers)
        pooling: Pooling strategy
        batch_size: Batch size for encoding
        device: Device for computation
        hessian_cache_dir: Cache directory for Hessian matrices
        use_emb_cache: Whether to use embedding cache
        emb_cache_dir: Embedding cache directory
        verbose: Verbosity level

    Returns:
        hessian_matrix: Array of shape (n_layers, n_layers)
    """

    # Check cache first
    os.makedirs(hessian_cache_dir, exist_ok=True)
    model_name_safe = model_name.replace("/", "_")
    task_name_safe = task_name.replace("/", "_")
    hessian_cache_path = os.path.join(
        hessian_cache_dir,
        f"{model_name_safe}_{task_name_safe}_hessian_{pooling}.pkl"
    )

    if os.path.exists(hessian_cache_path):
        if verbose >= 1:
            print(f"\nLoading cached Hessian matrix from {hessian_cache_path}")
        with open(hessian_cache_path, 'rb') as f:
            hessian_data = pickle.load(f)
        return hessian_data['hessian']

    if task is None:
        return None
    if verbose >= 1:
        print(f"\nCalculating Hessian matrix for {task_name}...")
        print(f"Evaluating ensemble at pairwise layer combinations...")

    # Convert layer_qualities to error (assuming higher quality = lower error)
    # Error = 1 - quality
    f_single = layer_qualities  # f(e_i) for each layer i
    #center point, equal weights
    encoder = AggregatedEncoder(
                    model_name=model_name,
                    pooling=pooling,
                    batch_size=batch_size,
                    device=device,
                    aggregation_weights=np.ones(n_layers) / n_layers,
                    use_cache=use_emb_cache,
                    cache_dir=emb_cache_dir
                )

    # Evaluate on MTEB task
    results = mteb.evaluate(
        model=encoder,
        tasks=[task],
        encode_kwargs={"batch_size": batch_size},
        show_progress_bar=False,
        overwrite_strategy="always"
    )
    # Extract quality score
    task_result = results[0]
    scores = task_result.scores[val_name][0]
    quality = scores.get('main_score', 0.0)
    f_zero = quality  # f(mean point) 

    # Evaluate at pairwise combinations: 0.5 * layer_i + 0.5 * layer_j
    pairwise_errors = np.zeros((n_layers, n_layers))

    # Suppress logging
    with warnings.catch_warnings():
        loggers_to_suppress = ['mteb', 'datasets', 'transformers', 'httpx', 'urllib3', 'filelock', 'huggingface_hub']
        original_levels = {}
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.INFO)

        total_pairs = n_layers * (n_layers + 1) // 2  # Including diagonal
        pair_count = 0

        for i in range(n_layers):
            for j in range(i, n_layers):
                pair_count += 1
                if verbose >= 1:
                    print(f"  Pair {pair_count}/{total_pairs}: layers ({i}, {j})...", end=" ", flush=True)

                # Create weight vector
                if i == j:
                    # Diagonal: evaluate at 0.5 * e_i
                    weights = np.zeros(n_layers)
                    weights[i] = 0.5
                else:
                    # Off-diagonal: evaluate at 0.5 * e_i + 0.5 * e_j
                    weights = np.zeros(n_layers)
                    weights[i] = 0.5
                    weights[j] = 0.5

                # Create aggregated encoder with these weights

                encoder = AggregatedEncoder(
                    model_name=model_name,
                    pooling=pooling,
                    batch_size=batch_size,
                    device=device,
                    aggregation_weights=weights,
                    use_cache=use_emb_cache,
                    cache_dir=emb_cache_dir
                )

                # Evaluate on MTEB task
                results = mteb.evaluate(
                    model=encoder,
                    tasks=[task],
                    encode_kwargs={"batch_size": batch_size},
                    show_progress_bar=False,
                    overwrite_strategy="always"
                )

                # Extract quality score
                task_result = results[0]
                scores = task_result.scores[val_name][0]
                quality = scores.get('main_score', 0.0)
                error = 1.0 - quality  # Convert to error

                pairwise_errors[i, j] = quality
                pairwise_errors[j, i] = quality  # Symmetric

                if verbose >= 1:
                    print(f"error: {quality:.4f}")

        # Restore logging levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)

    # Calculate Hessian using finite differences
    hessian = np.zeros((n_layers, n_layers))
    h = 0.5  # Step size

    for i in range(n_layers):
        for j in range(i, n_layers):
            if i == j:
                # Diagonal element: H_ii = [f(e_i) - 2*f(0.5*e_i) + f(0)] / h²
                H_ii = (f_single[i] - 2 * pairwise_errors[i, i] + f_zero) / (h ** 2)
                hessian[i, i] = H_ii
            else:
                # Off-diagonal element: H_ij = [f(0.5*e_i + 0.5*e_j) - f(e_i) - f(e_j) + f(0)] / (h * h)
                H_ij = (pairwise_errors[i, j] - f_single[i] - f_single[j] + f_zero) / (h * h)
                hessian[i, j] = H_ij
                hessian[j, i] = H_ij  # Symmetric

    # Cache the Hessian matrix
    cache_data = {
        'hessian': hessian,
        'pairwise_errors': pairwise_errors,
        'task_name': task_name,
        'model_name': model_name,
        'pooling': pooling
    }

    with open(hessian_cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    if verbose >= 1:
        print(f"\nCached Hessian matrix to {hessian_cache_path}")
        print(f"Hessian matrix shape: {hessian.shape}")
        print(f"Hessian stats: mean={hessian.mean():.4f}, std={hessian.std():.4f}")
        print(f"Hessian diagonal: {np.diag(hessian)}")

    return hessian


def calculate_matrix_least_squares(
    model_name: str,
    task,
    task_name: str,
    dataset_to_use: Dict,
    val_name: str,
    n_layers: int,
    layer_qualities: np.ndarray,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    hessian_cache_dir: str = "./hessian_cache",
    use_emb_cache: bool = True,
    emb_cache_dir: str = "./embs_cache",
    verbose: int = 1,
    num_random_points: int = 0  # Additional random evaluation points
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate Hessian matrix by fitting a second-order polynomial to ensemble quality.
    
    Fits: quality = c + w·coeff + w^T·H·w
    where w are normalized layer weights (sum to 1)
    
    Evaluation points:
    - Individual layers: e_i = (0,...,1,...,0) [n points]
    - Pairwise: 0.5*e_i + 0.5*e_j [n(n-1)/2 points]
    - Uniform: (1/n,...,1/n) [1 point]
    - Random: additional random simplex points [num_random_points]
    
    Args:
        ... (same as before)
        num_random_points: Number of additional random evaluation points
        
    Returns:
        c: Constant term (baseline quality)
        coeff: Linear coefficients, shape (n_layers,)
        H: Hessian matrix, shape (n_layers, n_layers)
    """
    
    # Check cache first
    os.makedirs(hessian_cache_dir, exist_ok=True)
    model_name_safe = model_name.replace("/", "_")
    task_name_safe = task_name.replace("/", "_")
    hessian_cache_path = os.path.join(
        hessian_cache_dir,
        f"{model_name_safe}_{task_name_safe}_hessian_lsq_{pooling}.pkl"
    )
    
    if os.path.exists(hessian_cache_path):
        if verbose >= 1:
            print(f"\nLoading cached Hessian from {hessian_cache_path}")
        with open(hessian_cache_path, 'rb') as f:
            cached = pickle.load(f)
        return cached['c'], cached['coeff'], cached['hessian']
    
    if task is None:
        return None, None, None
    
    if verbose >= 1:
        print(f"\nCalculating Hessian via least-squares fitting for {task_name}...")
    
    # Collect evaluation points and their qualities
    eval_points = []  # List of weight vectors
    eval_qualities = []  # List of quality scores
    
    # Suppress logging
    with warnings.catch_warnings():
        loggers_to_suppress = ['mteb', 'datasets', 'transformers', 'httpx', 'urllib3', 'filelock', 'huggingface_hub']
        original_levels = {}
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.INFO)
        
        # 1. Individual layers (already computed - use layer_qualities)
        if verbose >= 1:
            print(f"\n1. Using individual layer qualities ({n_layers} points)")
        
        for i in range(n_layers):
            weights = np.zeros(n_layers)
            weights[i] = 1.0
            eval_points.append(weights)
            eval_qualities.append(layer_qualities[i])
        
        # 2. Uniform weights
        if verbose >= 1:
            print(f"2. Evaluating uniform weights (1 point)")
        
        uniform_weights = np.ones(n_layers) / n_layers
        encoder = AggregatedEncoder(
            model_name=model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            aggregation_weights=uniform_weights,
            use_cache=use_emb_cache,
            cache_dir=emb_cache_dir
        )
        results = mteb.evaluate(
            model=encoder,
            tasks=[task],
            encode_kwargs={"batch_size": batch_size},
            show_progress_bar=False,
            overwrite_strategy="always"
        )
        task_result = results[0]
        scores = task_result.scores[val_name][0]
        uniform_quality = scores.get('main_score', 0.0)
        
        eval_points.append(uniform_weights)
        eval_qualities.append(uniform_quality)
        
        if verbose >= 1:
            print(f"   Uniform quality: {uniform_quality:.4f}")
        
        # 3. Pairwise combinations: normalized (0.5, 0.5) pairs
        num_pairs = n_layers * (n_layers - 1) // 2
        if verbose >= 1:
            print(f"3. Evaluating pairwise combinations ({num_pairs} points)")
        
        pair_count = 0
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                pair_count += 1
                if verbose >= 1:
                    print(f"   Pair {pair_count}/{num_pairs}: layers ({i}, {j})...", end=" ", flush=True)
                
                # Create normalized weight vector
                weights = np.zeros(n_layers)
                weights[i] = 0.5
                weights[j] = 0.5
                # Already sums to 1!
                
                encoder = AggregatedEncoder(
                    model_name=model_name,
                    pooling=pooling,
                    batch_size=batch_size,
                    device=device,
                    aggregation_weights=weights,
                    use_cache=use_emb_cache,
                    cache_dir=emb_cache_dir
                )
                
                results = mteb.evaluate(
                    model=encoder,
                    tasks=[task],
                    encode_kwargs={"batch_size": batch_size},
                    show_progress_bar=False,
                    overwrite_strategy="always"
                )
                
                task_result = results[0]
                scores = task_result.scores[val_name][0]
                quality = scores.get('main_score', 0.0)
                
                eval_points.append(weights)
                eval_qualities.append(quality)
                
                if verbose >= 1:
                    print(f"quality: {quality:.4f}")
        
        # 4. Random simplex points (optional)
        if num_random_points > 0:
            if verbose >= 1:
                print(f"4. Evaluating random simplex points ({num_random_points} points)")
            
            np.random.seed(42)
            for r in range(num_random_points):
                # Sample from simplex using Dirichlet distribution
                weights = np.random.dirichlet(np.ones(n_layers))
                
                if verbose >= 1:
                    print(f"   Random {r+1}/{num_random_points}...", end=" ", flush=True)
                
                encoder = AggregatedEncoder(
                    model_name=model_name,
                    pooling=pooling,
                    batch_size=batch_size,
                    device=device,
                    aggregation_weights=weights,
                    use_cache=use_emb_cache,
                    cache_dir=emb_cache_dir
                )
                
                results = mteb.evaluate(
                    model=encoder,
                    tasks=[task],
                    encode_kwargs={"batch_size": batch_size},
                    show_progress_bar=False,
                    overwrite_strategy="always"
                )
                
                task_result = results[0]
                scores = task_result.scores[val_name][0]
                quality = scores.get('main_score', 0.0)
                
                eval_points.append(weights)
                eval_qualities.append(quality)
                
                if verbose >= 1:
                    print(f"quality: {quality:.4f}")
        
        # Restore logging levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)
    
    # Convert to numpy arrays
    eval_points = np.array(eval_points)  # Shape: (num_points, n_layers)
    eval_qualities = np.array(eval_qualities)  # Shape: (num_points,)
    
    num_points = len(eval_points)
    if verbose >= 1:
        print(f"\nFitting quadratic model to {num_points} evaluation points...")
        print(f"Quality range: [{eval_qualities.min():.4f}, {eval_qualities.max():.4f}]")
    
    # Build design matrix for least squares
    # Model: quality = c + sum_i(coeff_i * w_i) + sum_i sum_j (H_ij * w_i * w_j)
    # Number of parameters: 1 (c) + n (coeff) + n(n+1)/2 (H, symmetric)
    
    num_params = 1 + n_layers + n_layers * (n_layers + 1) // 2
    X = np.zeros((num_points, num_params))
    
    for point_idx in range(num_points):
        w = eval_points[point_idx]
        col_idx = 0
        
        # Constant term
        X[point_idx, col_idx] = 1.0
        col_idx += 1
        
        # Linear terms: w_i
        for i in range(n_layers):
            X[point_idx, col_idx] = w[i]
            col_idx += 1
        
        # Quadratic terms: w_i * w_j (symmetric, store upper triangle)
        for i in range(n_layers):
            for j in range(i, n_layers):
                X[point_idx, col_idx] = w[i] * w[j]
                col_idx += 1
    
    # Solve least squares: X * params = eval_qualities
    params, residuals, rank, s = np.linalg.lstsq(X, eval_qualities, rcond=None)
    
    # Extract parameters
    col_idx = 0
    
    # Constant
    c = params[col_idx]
    col_idx += 1
    
    # Linear coefficients
    coeff = params[col_idx:col_idx + n_layers]
    col_idx += n_layers
    
    # Hessian (symmetric)
    H = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(i, n_layers):
            H[i, j] = params[col_idx]
            H[j, i] = params[col_idx]
            col_idx += 1
    
    # Compute R² and RMSE for diagnostics
    predictions = X @ params
    ss_res = np.sum((eval_qualities - predictions) ** 2)
    ss_tot = np.sum((eval_qualities - eval_qualities.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / num_points)
    
    if verbose >= 1:
        print(f"\nLeast squares fitting results:")
        print(f"  R² = {r_squared:.6f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  Constant c = {c:.6f}")
        print(f"  Linear coeff range: [{coeff.min():.6f}, {coeff.max():.6f}]")
        print(f"  Hessian H range: [{H.min():.6f}, {H.max():.6f}]")
        print(f"  Hessian diagonal: {np.diag(H)}")
    
    # Cache the results
    cache_data = {
        'c': c,
        'coeff': coeff,
        'hessian': H,
        'eval_points': eval_points,
        'eval_qualities': eval_qualities,
        'r_squared': r_squared,
        'rmse': rmse,
        'task_name': task_name,
        'model_name': model_name,
        'pooling': pooling
    }
    
    with open(hessian_cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    if verbose >= 1:
        print(f"\nCached Hessian to {hessian_cache_path}")
    
    return c, coeff, H


def calculate_hessian_matrix_fisher(
    model_name: str,
    task,
    task_name: str,
    dataset_to_use: Dict,
    val_name: str,
    n_layers: int,
    layer_qualities: np.ndarray,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    hessian_cache_dir: str = "./hessian_cache",
    use_emb_cache: bool = True,
    emb_cache_dir: str = "./embs_cache",
    verbose: int = 1,
    perturbation_size: float = 0.1,
    num_samples: int = 1000
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate TRUE Fisher Information Matrix using per-sample gradients.
    
    Fisher approximation:
    1. Evaluate at center (best layer)
    2. Get per-sample scores at center
    3. For each layer, perturb and get per-sample scores
    4. Compute per-sample gradients: ∂log L(x)/∂w_i for each sample x
    5. Fisher = Covariance of per-sample gradients across samples
    6. Hessian ≈ -Fisher (for maximization)
    
    Cost: O(n) evaluations + per-sample prediction parsing
    
    Returns:
        c: Log-likelihood at center
        coeff: Average gradient (linear coefficient)
        H: Hessian = -Fisher
    """
    
    # Check cache
    os.makedirs(hessian_cache_dir, exist_ok=True)
    model_name_safe = model_name.replace("/", "_")
    task_name_safe = task_name.replace("/", "_")
    hessian_cache_path = os.path.join(
        hessian_cache_dir,
        f"{model_name_safe}_{task_name_safe}_hessian_fisher_persample_{pooling}.pkl"
    )
    
    if os.path.exists(hessian_cache_path):
        if verbose >= 1:
            print(f"\nLoading cached Fisher-Hessian (per-sample) from {hessian_cache_path}")
        with open(hessian_cache_path, 'rb') as f:
            cached = pickle.load(f)
        return cached['c'], cached['coeff'], cached['hessian']
    
    if task is None:
        return None, None, None
    
    if verbose >= 1:
        print(f"\n{'='*70}")
        print(f"TRUE Fisher Information Matrix (per-sample gradients)")
        print(f"{'='*70}")
    
    # Find best layer as center
    best_layer_idx = np.argmax(layer_qualities)
    center_weights = np.zeros(n_layers)
    center_weights[best_layer_idx] = 1.0
    
    if verbose >= 1:
        print(f"Center: best layer {best_layer_idx} (quality: {layer_qualities[best_layer_idx]:.4f})")
    
    # Create prediction directory
    prediction_base_dir = os.path.join(hessian_cache_dir, "fisher_predictions")
    os.makedirs(prediction_base_dir, exist_ok=True)
    
    # Suppress logging
    with warnings.catch_warnings():
        loggers_to_suppress = ['mteb', 'datasets', 'transformers', 'httpx', 
                               'urllib3', 'filelock', 'huggingface_hub']
        original_levels = {}
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.INFO)
        
        # Step 1: Evaluate at center and get per-sample scores
        if verbose >= 1:
            print(f"\n1. Evaluating at center (best layer {best_layer_idx})...")
        
        encoder_center = AggregatedEncoder(
            model_name=model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            aggregation_weights=center_weights,
            use_cache=use_emb_cache,
            cache_dir=emb_cache_dir
        )
        
        prediction_dir_center = os.path.join(prediction_base_dir, "center")
        os.makedirs(prediction_dir_center, exist_ok=True)
        # In calculate_hessian_matrix_fisher, when calling mteb.evaluate:
        print(encoder_center,task,batch_size, prediction_dir_center)
        results_center = mteb.evaluate(
            model=encoder_center,
            tasks=[task],
            encode_kwargs={"batch_size": batch_size},
            prediction_folder=prediction_dir_center,
            show_progress_bar=False,
            overwrite_strategy="always"
        )
        
        task_result = results_center[0]
        scores = task_result.scores[val_name][0]
        quality_center = scores.get('main_score', 0.0)
        
        # Extract per-sample scores from predictions
        per_sample_scores_center = load_per_sample_scores_from_predictions(
            prediction_dir_center, task, task_name, val_name, dataset_to_use
        )
        
        if verbose >= 1:
            print(f"   Aggregate quality: {quality_center:.4f}")
            print(f"   Per-sample scores extracted: {len(per_sample_scores_center)} samples")
            print(f"   Per-sample score range: [{per_sample_scores_center.min():.4f}, {per_sample_scores_center.max():.4f}]")
        
        # Subsample if needed
        num_samples_actual = len(per_sample_scores_center)
        # if num_samples > 0 and num_samples < num_samples_actual:
        #     indices = np.random.choice(num_samples_actual, num_samples, replace=False)
        #     per_sample_scores_center = per_sample_scores_center[indices]
        #     if verbose >= 1:
        #         print(f"   Subsampled to {num_samples} samples")
        # else:
        num_samples = num_samples_actual
        
        # Normalize per-sample scores to probabilities
        per_sample_probs_center = per_sample_scores_center #/ per_sample_scores_center.sum()
        per_sample_log_L_center = np.log(np.maximum(per_sample_probs_center, 1e-10))
        print(per_sample_log_L_center)
        c = np.mean(per_sample_log_L_center)  # Average log-likelihood
        
        if verbose >= 1:
            print(f"   Average log-likelihood: {c:.4f}")
        
        # Step 2: For each layer, perturb and compute per-sample gradients
        if verbose >= 1:
            print(f"\n2. Computing per-sample gradients ({n_layers} perturbations):")
        
        per_sample_gradients = []  # Will be shape (n_layers, num_samples)
        
        for i in range(n_layers):
            if verbose >= 1:
                print(f"   [{i+1}/{n_layers}] Layer {i}...", end=" ", flush=True)
            
            if i == best_layer_idx:
                # At center - gradient is zero
                gradient_i = np.zeros(num_samples)
                per_sample_gradients.append(gradient_i)
                if verbose >= 1:
                    print(f"(at center, gradient=0)")
                continue
            
            # Perturb toward layer i
            target_weights = np.zeros(n_layers)
            target_weights[i] = 1.0
            perturbed_weights = (1 - perturbation_size) * center_weights + \
                               perturbation_size * target_weights
            
            # Evaluate
            encoder_i = AggregatedEncoder(
                model_name=model_name,
                pooling=pooling,
                batch_size=batch_size,
                device=device,
                aggregation_weights=perturbed_weights,
                use_cache=use_emb_cache,
                cache_dir=emb_cache_dir
            )
            
            prediction_dir_i = os.path.join(prediction_base_dir, f"layer_{i}")
            os.makedirs(prediction_dir_i, exist_ok=True)
            
            results_i = mteb.evaluate(
                model=encoder_i,
                tasks=[task],
                encode_kwargs={"batch_size": batch_size},
                prediction_folder=prediction_dir_i,
                show_progress_bar=False,
                overwrite_strategy="always"
            )
            
            # Extract per-sample scores
            per_sample_scores_i = load_per_sample_scores_from_predictions(
                prediction_dir_i, task, task_name, val_name, dataset_to_use
            )
            
            # if num_samples < num_samples_actual:
            #     per_sample_scores_i = per_sample_scores_i[indices]
            
            # Normalize and take log
            per_sample_probs_i = per_sample_scores_i #/ per_sample_scores_i.sum()
            per_sample_log_L_i = np.log(np.maximum(per_sample_probs_i, 1e-10))
            
            # Per-sample gradient for dimension i
            print(per_sample_log_L_i)
            gradient_i = (per_sample_log_L_i - per_sample_log_L_center) / perturbation_size
            per_sample_gradients.append(gradient_i)
            
            if verbose >= 1:
                mean_grad = gradient_i.mean()
                std_grad = gradient_i.std()
                print(f"mean={mean_grad:+.10f}, std={std_grad:.10f}, non-zero: {(gradient_i != 0).mean()}")
                print(gradient_i)
        
        # Restore logging
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)
    
    # Step 3: Compute Fisher Information Matrix
    # Shape: (n_layers, num_samples)
    per_sample_gradients = np.array(per_sample_gradients)
    
    if verbose >= 1:
        print(f"\n3. Computing Fisher Information Matrix:")
        print(f"   Per-sample gradients shape: {per_sample_gradients.shape}")
    
    # Linear coefficient: average gradient across samples
    coeff = per_sample_gradients.sum(axis=1)  # Shape: (n_layers,) #avoid normalization
    print(coeff)
    # Fisher = Covariance of gradients across samples
    # F_ij = Cov[∂log L(x)/∂w_i, ∂log L(x)/∂w_j]
    F = np.cov(per_sample_gradients)  # Shape: (n_layers, n_layers)
    print(F)
    # Hessian for maximization: H = -F
    H = -F
    
    # Ensure symmetry
    H = (H + H.T) / 2
    
    # Diagnostics
    eigenvalues_F = np.linalg.eigvals(F)
    eigenvalues_H = np.linalg.eigvals(H)
    
    if verbose >= 1:
        print(f"\n{'='*70}")
        print(f"Fisher Information Matrix Results:")
        print(f"{'='*70}")
        print(f"  Average log-likelihood: {c:.6f}")
        print(f"  Linear coefficient (coeff) norm: {np.linalg.norm(coeff):.6f}")
        print(f"  Linear coefficient range: [{coeff.min():.6f}, {coeff.max():.6f}]")
        print(f"\nFisher Matrix F:")
        print(f"  F diagonal: {np.diag(F)}")
        print(f"  F eigenvalue range: [{eigenvalues_F.min():.6f}, {eigenvalues_F.max():.6f}]")
        print(f"  F trace: {np.trace(F):.6f}")
        print(f"\nHessian H = -F:")
        print(f"  H diagonal: {np.diag(H)}")
        print(f"  H eigenvalue range: [{eigenvalues_H.min():.6f}, {eigenvalues_H.max():.6f}]")
        
        if np.all(eigenvalues_H < 1e-6):
            print(f"  ✓ Hessian is negative definite (local maximum)")
        elif np.all(eigenvalues_H > -1e-6):
            print(f"  ⚠ Hessian is positive definite (local minimum)")
        else:
            print(f"  ⚠ Hessian is indefinite (saddle point)")
        print(f"{'='*70}\n")
    
    # Cache
    cache_data = {
        'c': c,
        'coeff': coeff,
        'hessian': H,
        'fisher': F,
        'per_sample_gradients': per_sample_gradients,
        'best_layer_idx': best_layer_idx,
        'num_samples': num_samples,
        'perturbation_size': perturbation_size,
        'method': 'fisher_per_sample',
        'task_name': task_name,
        'model_name': model_name,
        'pooling': pooling
    }
    
    with open(hessian_cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    if verbose >= 1:
        print(f"✓ Cached Fisher-Hessian to {hessian_cache_path}\n")
    
    return center_weights, coeff, H


def load_per_sample_scores_from_predictions(
    prediction_dir: str,
    task_resolver
) -> np.ndarray:
    """
    Load per-sample scores from MTEB prediction files.
    Returns array of per-sample scores. NO FALLBACK SCORES - raises errors!
    """
    import json
    from pathlib import Path
    
    # Find prediction file
    pred_dir_path = Path(prediction_dir)
    if not pred_dir_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    
    pred_files = list(pred_dir_path.glob("*.json"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files in {prediction_dir}")
    
    pred_file = pred_files[0]
    
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    # All nesting logic delegated to resolver
    split_predictions = task_resolver.unwrap_predictions(pred_data)

    if isinstance(split_predictions, list) and split_predictions and isinstance(split_predictions[0], list):
        split_predictions = split_predictions[0]

    val_data = task_resolver.dataset.get(task_resolver.val_name)
    if val_data is None:
        raise ValueError(f"Validation split '{task_resolver.val_name}' not found in dataset")
    
    # Extract per-sample scores based on task type
    if task_resolver.task_type == "Classification":
        # Per-sample correctness
        if 'label' in val_data.column_names:
            true_labels = list(val_data['label'])
        elif 'labels' in val_data.column_names:
            true_labels = list(val_data['labels'])
        else:
            raise ValueError(f"No label column. Available: {val_data.column_names}")
        true_labels = list(true_labels)
        if type(true_labels[0]) is list:
            true_labels = true_labels[0]
        print("split_predictions", split_predictions)
        print("true_labels", true_labels)
        # Predictions format check
        if isinstance(split_predictions, dict):
            pred_list = [split_predictions.get(str(i), split_predictions.get(i)) for i in range(len(true_labels))]
            if None in pred_list:
                raise ValueError(f"Missing predictions for some samples")
        elif isinstance(split_predictions, list):
            pred_list = split_predictions
        else:
            raise ValueError(f"Unexpected predictions format: {type(split_predictions)}")
        
        per_sample_scores = np.array([1.0 if pred == true else 0.0
                                     for pred, true in zip(pred_list, true_labels)])
    
    elif task_resolver.task_type == "STS":
        # Per-pair score based on prediction error
        if 'score' not in val_data.column_names:
            raise ValueError(f"No 'score' column. Available: {val_data.column_names}")
        
        true_scores = np.array(val_data['score'])
        
        if isinstance(split_predictions, list):
            pred_scores = np.array(split_predictions)
        elif isinstance(split_predictions, dict):
            pred_scores = np.array([split_predictions[str(i)] if str(i) in split_predictions else split_predictions[i] 
                                   for i in range(len(true_scores))])
        else:
            raise ValueError(f"Unexpected predictions format: {type(split_predictions)}")
        
        # Normalize to [0, 5] range (cosine similarity [-1,1] -> [0,5])
        pred_scores_normalized = (pred_scores + 1) * 2.5
        absolute_errors = np.abs(pred_scores_normalized - true_scores)
        per_sample_scores = np.maximum(1.0 - (absolute_errors / 5.0), 0.0)
    
    elif task_resolver.task_type == "Retrieval":
        # Per-query binary relevance or NDCG
        if 'queries' not in val_data:
            raise ValueError("No 'queries' in validation data")
        
        queries = val_data['queries']
        qrels = val_data.get('qrels') or val_data.get('relevant_docs')
        if qrels is None:
            raise ValueError("No 'qrels' or 'relevant_docs' in validation data")
        
        if isinstance(queries, dict):
            query_ids = list(queries.keys())
        else:
            raise ValueError(f"Unexpected queries format: {type(queries)}")
        
        # split_predictions should be dict of {query_id: {doc_id: score}}
        if not isinstance(split_predictions, dict):
            raise ValueError(f"Expected dict for retrieval predictions, got {type(split_predictions)}")
        
        per_sample_scores = []
        for qid in query_ids:
            qid_str = str(qid)
            
            if qid_str not in split_predictions:
                raise ValueError(f"Query {qid_str} not in predictions")
            if qid_str not in qrels:
                raise ValueError(f"Query {qid_str} not in qrels")
            
            # Get top-10 retrieved docs
            if isinstance(split_predictions[qid_str], dict):
                # Sort by score descending
                retrieved_docs = sorted(split_predictions[qid_str].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_docs]
            elif isinstance(split_predictions[qid_str], list):
                retrieved_doc_ids = split_predictions[qid_str][:10]
            else:
                raise ValueError(f"Unexpected predictions format for query {qid_str}")
            
            if isinstance(qrels[qid_str], dict):
                relevant_docs = set(qrels[qid_str].keys())
            elif isinstance(qrels[qid_str], list):
                relevant_docs = set(qrels[qid_str])
            else:
                raise ValueError(f"Unexpected qrels format for query {qid_str}")
            
            # Binary: any relevant in top-10?
            has_relevant = any(doc_id in relevant_docs for doc_id in retrieved_doc_ids)
            per_sample_scores.append(1.0 if has_relevant else 0.0)
        
        per_sample_scores = np.array(per_sample_scores)
    
    elif task_resolver.task_type == "PairClassification":
        # Per-pair correctness
        if 'label' not in val_data.column_names:
            raise ValueError(f"No 'label' column. Available: {val_data.column_names}")
        
        true_labels = val_data['label']
        
        if isinstance(split_predictions, dict):
            pred_list = [split_predictions[str(i)] if str(i) in split_predictions else split_predictions[i] 
                        for i in range(len(true_labels))]
        elif isinstance(split_predictions, list):
            pred_list = split_predictions
        else:
            raise ValueError(f"Unexpected predictions format: {type(split_predictions)}")
        
        per_sample_scores = np.array([1.0 if pred == true else 0.0 
                                     for pred, true in zip(pred_list, true_labels)])
    
    elif task_resolver.task_type == "Clustering":
        # Per-sample cluster purity
        if 'label' not in val_data.column_names:
            raise ValueError(f"No 'label' column for clustering")
        
        true_labels = np.array(val_data['label'])
        
        if isinstance(split_predictions, list):
            pred_clusters = np.array(split_predictions)
        elif isinstance(split_predictions, dict):
            pred_clusters = np.array([split_predictions[str(i)] if str(i) in split_predictions else split_predictions[i] 
                                     for i in range(len(true_labels))])
        else:
            raise ValueError(f"Unexpected predictions format: {type(split_predictions)}")
        
        per_sample_scores = []
        for i, (pred_cluster, true_label) in enumerate(zip(pred_clusters, true_labels)):
            same_cluster_mask = (pred_clusters == pred_cluster)
            same_cluster_true_labels = true_labels[same_cluster_mask]
            
            if len(same_cluster_true_labels) > 0:
                purity = np.sum(same_cluster_true_labels == true_label) / len(same_cluster_true_labels)
            else:
                purity = 0.0
            
            per_sample_scores.append(max(purity, 0.01))
        
        per_sample_scores = np.array(per_sample_scores)
    
    elif task_resolver.task_type == "Reranking":
        # Similar to retrieval
        if 'queries' not in val_data:
            raise ValueError("No 'queries' in validation data")
        
        queries = val_data['queries']
        qrels = val_data.get('qrels') or val_data.get('relevant_docs')
        if qrels is None:
            raise ValueError("No 'qrels' in validation data")
        
        if not isinstance(split_predictions, dict):
            raise ValueError(f"Expected dict for reranking predictions")
        
        query_ids = list(queries.keys()) if isinstance(queries, dict) else queries
        
        per_sample_scores = []
        for qid in query_ids:
            qid_str = str(qid)
            
            if qid_str not in split_predictions or qid_str not in qrels:
                raise ValueError(f"Query {qid_str} missing from predictions or qrels")
            
            # Binary relevance
            if isinstance(split_predictions[qid_str], dict):
                retrieved_docs = sorted(split_predictions[qid_str].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_docs]
            else:
                retrieved_doc_ids = split_predictions[qid_str][:10]
            
            relevant_docs = set(qrels[qid_str].keys() if isinstance(qrels[qid_str], dict) else qrels[qid_str])
            
            has_relevant = any(doc_id in relevant_docs for doc_id in retrieved_doc_ids)
            per_sample_scores.append(1.0 if has_relevant else 0.01)
        
        per_sample_scores = np.array(per_sample_scores)
    
    else:
        raise ValueError(f"Unsupported task type: {task_resolver.task_type}")
    
    if len(per_sample_scores) == 0:
        raise ValueError("No per-sample scores were computed!")
    
    return per_sample_scores
