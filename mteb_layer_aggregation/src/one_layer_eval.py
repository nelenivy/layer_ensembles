from typing import List, Tuple, Optional, Union, Dict
import numpy as np
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

def split_retrieval_data(queries, corpus, qrels, val_size=0.2, seed=42):
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
            'qrels': train_qrels
        },
        'test': {
            'queries': val_queries,
            'corpus': corpus,
            'qrels': val_qrels
        }
    }

class SingleLayerEncoder(EncoderProtocol):
    """Single layer encoder compatible with MTEB 2.0"""
    
    def __init__(self, model, tokenizer, layer_idx, pooling, device, batch_size, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = model.config.hidden_size
        self.model_name = f"{model_name}_layer{layer_idx}"
    
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
    
    def _encode_batch(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Encode a single batch of sentences."""
        if not sentences:
            return None
        
        # Filter empty strings
        sentences = [str(s).strip() for s in sentences if s is not None]
        sentences = [s for s in sentences if s]
        if not sentences:
            return None
        
        try:
            inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states[self.layer_idx]
                
                # Apply pooling
                if self.pooling == "cls":
                    pooled = hidden_states[:, 0, :]
                else:  # mean
                    mask = inputs['attention_mask'].unsqueeze(-1).float()
                    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                
                return pooled.cpu().numpy()
        except Exception as e:
            print(f"Error encoding batch: {e}")
            return None
    
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
        
            
def compute_dataset_specific_layer_quality(
    model_name: str,
    task_name: str,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "cuda",
    verbose: int = 1
) -> np.ndarray:
    """
    Compute layer quality by evaluating each layer on a specific MTEB task.
    This implements the notebook approach where validation quality is computed
    for each dataset individually.
    
    Args:
        model_name: HuggingFace model name
        task_name: MTEB task name (e.g., "Banking77Classification")
        pooling: Pooling strategy ('mean' or 'cls')
        batch_size: Batch size for encoding
        device: Device for computation
        verbose: Verbosity level (0, 1, 2)
        
    Returns:
        layer_qualities: Array of shape (n_layers,) with quality scores
    """
    
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

    # Loads the dataset from HuggingFace
    dataset_to_use = task.dataset['default'] if 'default' in task.dataset else task.dataset
    val_name = None
    
    for split_name in val_splits:
        if split_name in dataset_to_use:
            val_name = split_name
            print("!!!!!!!!!!!!!!!!!! found!!!!!!!!!!!!!!!!!")

    if val_name is None:
        print("!!!!!!!!!!!!!!!!!!NOT found!!!!!!!!!!!!!!!!!")
        print(list(dataset_to_use.keys()))
        # Monkey-patch the task's dataset

        if type(dataset_to_use['train']) is dict:
            #retrieval tasks
            # Then use the split function above
            trainval = split_retrieval_data(
                dataset_to_use['train']['queries'],
                dataset_to_use['train']['corpus'],
                dataset_to_use['train']['qrels']
            )
        else:
            trainval = dataset_to_use['train'].train_test_split(test_size=0.2, seed=42)
            
        dataset_to_use['train'] = trainval['train']
        dataset_to_use['validation'] = trainval['test']
        val_name = 'validation'
          
    dataset_to_use = task.dataset['default'] if 'default' in task.dataset else task.dataset
    print(list(dataset_to_use.keys()))
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
                model_name=model_name
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
    
    return layer_qualities
