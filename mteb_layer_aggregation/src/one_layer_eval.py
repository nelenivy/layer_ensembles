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
from datasets import disable_caching
import random
from collections import defaultdict
from src.cache_manager import QualityCache, EmbeddingCache
from src.metrics.calculator import SimilarityCalculator, AVAILABLE_METRICS
from collections.abc import Iterable
from datasets.features.features import Value  # HuggingFace column types

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
    
    def _encode_batch(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Encode batch with sentence-level caching"""
        if not sentences:
            return None
        
        # Normalize sentences
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
            sentences = [sentences]
        
        # Handle list of sentences
        if not sentences or len(sentences) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
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
        layer_embs = encoder._encode_batch(sentences)
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
        # Limit samples for similarity calculation
        if len(layer_embs) > similarity_sample_size * 2:
            layer_embs = layer_embs[:similarity_sample_size * 2]
        
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
    use_emb_cache=True, 
    emb_cache_dir="./embs_cache",
    calc_similarity_matrix: bool = False,
    similarity_metric: Optional[str] = None,
    similarity_sample_size: int = 3000,
    similarity_num_trials: int = 5,
    similarity_cache_dir: str = "./similarity_cache"
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute layer quality by evaluating each layer on a specific MTEB task with caching support.
    Optionally calculate similarity matrix using the same task/split data.
    
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
        
    Returns:
        layer_qualities: Array of shape (n_layers,) with quality scores
        OR
        (layer_qualities, similarity_matrix): Tuple if calc_similarity_matrix=True
    """

    # Check cache first for quality
    cached_quality = None
    if use_cache:
        quality_cache = QualityCache(cache_dir)
        cached_quality = quality_cache.get(model_name, pooling, task_name)
        if cached_quality is not None:
            # If we also need similarity matrix, check its cache
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
            else:
                return cached_quality

    if verbose >= 1:
        print(f"\n{'='*70}")
        print(f"Computing layer quality on: {task_name}")
        print(f"{'='*70}\n")
    
    # Load task
    val_splits = ['dev', 'validation', 'val']
    # Disable caching globally
    disable_caching()
    task = None
    for split_name in val_splits:
        try:
            print(split_name)
            print(1)
            task = mteb.get_task(task_name, languages=['eng'], eval_splits=[split_name], exclusive_language_filter=True)
            print(2)
            task.load_data()
            print(3)
            break
        except:
            task = None
            
    if task is None:
        task = mteb.get_task(task_name, languages=['eng'], eval_splits=['train'])
        task.load_data()
    task_type = task.metadata.type
    # Loads the dataset from HuggingFace
    dataset_to_use = task.dataset['default'] if 'default' in task.dataset else task.dataset
    dataset_to_use = dataset_to_use['en'] if 'en' in dataset_to_use else dataset_to_use
    dataset_to_use = dataset_to_use['default'] if 'default' in dataset_to_use else dataset_to_use
    val_name = None
    
    for split_name in val_splits:
        if split_name in dataset_to_use:
            val_name = split_name
            print("!!!!!!!!!!!!!!!!!! found!!!!!!!!!!!!!!!!!")

    if val_name is None:
        print("!!!!!!!!!!!!!!!!!!NOT found!!!!!!!!!!!!!!!!!")
        print(list(dataset_to_use.keys()))
        # Monkey-patch the task's dataset
        print(task_type)
        if task_type == "Classification":
            print('for classification')
            trainval = dataset_to_use['train'].train_test_split(test_size=0.2, seed=42)
            dataset_to_use['train'] = trainval['train']
            dataset_to_use['validation'] = trainval['test']
            val_name = 'validation'
        else:
            val_name = 'train'
        # if type(dataset_to_use['train']) is dict:
        #     print(list(dataset_to_use['train'].keys()))
        #     #retrieval tasks
        #     # Determine qrels key name
        #     qrels_key = None
        #     if 'qrels' in dataset_to_use['train']:
        #         qrels_key = 'qrels'
        #     elif 'relevant_docs' in dataset_to_use['train']:
        #         qrels_key = 'relevant_docs'
        #     else:
        #         raise ValueError(f"Cannot find qrels/relevant_docs in {keys}")
        #     # Then use the split function above
        #     trainval = split_retrieval_data(
        #         dataset_to_use['train']['queries'],
        #         dataset_to_use['train']['corpus'],
        #         dataset_to_use['train'][qrels_key],
        #         qrels_key=qrels_key
        #     )
                
    task.__dict__['_eval_splits'] = [val_name]
    # SIMPLE FIX: Set n_experiments directly if it's a classification task
    if hasattr(task, 'n_experiments'):
        if verbose >= 2:
            print(f"  Setting n_experiments={1} (was {task.n_experiments})")
        task.n_experiments = 1
    # Load model once for all layers
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    ).to(device).eval()
    
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    layer_qualities = []
    
    if verbose >= 1:
        print(f"Evaluating {n_layers} layers...")
    # Temporarily suppress MTEB logging
    # Suppress warnings
    if cached_quality is None:
        with warnings.catch_warnings():
            #warnings.filterwarnings('ignore', category=FutureWarning)
            #warnings.filterwarnings('ignore', category=UserWarning)
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
                #try:
                results = mteb.evaluate(
                    model=encoder,
                    #eval_splits=[val_name],
                    tasks=[task],
                    # Remove eval_splits - it's passed via task or results extraction
                    encode_kwargs={"batch_size": batch_size},
                    show_progress_bar=False,
                    overwrite_strategy="always"  #'only-missing'
                )

                # Extract main score for the requested split
                # results is a list of TaskResult objects
                task_result = results[0]

                # Try to get the requested split, fallback to available splits
                scores = task_result.scores[val_name][0]
                quality = scores.get('main_score', 0.0)
                layer_qualities.append(quality)
                
                if verbose >= 1:
                    print(f"{quality:.4f}")
                        
                # except Exception as e:
                #     if verbose >= 1:
                #         print(f"ERROR: {e}")
                #     layer_qualities.append(0.0)
            
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
        
    # Calculate similarity matrix if requested
    # At this point we have: model, tokenizer, task, dataset_to_use, val_name, n_layers
    if calc_similarity_matrix and similarity_metric:
        similarity_matrix = calculate_similarity_matrix_from_task(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            task_name=task_name,
            dataset_to_use=dataset_to_use,
            val_name=val_name,
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

    return layer_qualities
