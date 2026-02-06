"""
Task-specific evaluators for layer quality assessment.
Provides evaluation methods for different task types before aggregation.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
from ..models.encoders import SimpleEncoder


def evaluate_layer_classification(
    layer_idx: int,
    model_name: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    pooling: str = "mean",
    max_iter: int = 500,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate single layer on classification task.

    Args:
        layer_idx: Layer index to evaluate
        model_name: HuggingFace model name
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        pooling: Pooling strategy
        max_iter: Max iterations for LogisticRegression
        batch_size: Batch size for encoding

    Returns:
        Dictionary with accuracy and metrics
    """
    encoder = SimpleEncoder(model_name, pooling=pooling, layer_idx=layer_idx, batch_size=batch_size)

    # Encode
    X_train = encoder.encode(train_texts).numpy()
    X_val = encoder.encode(val_texts).numpy()

    # Train classifier
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(X_train, train_labels)

    # Evaluate
    preds = clf.predict(X_val)
    acc = accuracy_score(val_labels, preds)
    f1 = f1_score(val_labels, preds, average='macro')

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "layer": layer_idx
    }


def evaluate_layer_retrieval(
    layer_idx: int,
    model_name: str,
    queries: List[Dict],
    corpus: List[Dict],
    relevance: List[Dict],
    pooling: str = "mean",
    top_k: int = 10,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate single layer on retrieval task using Recall@k.

    Args:
        layer_idx: Layer index
        model_name: Model name
        queries: List of query dicts with '_id' and 'text'
        corpus: List of corpus dicts with '_id' and 'text'
        relevance: List of relevance judgments
        pooling: Pooling strategy
        top_k: Top-k for recall
        batch_size: Batch size

    Returns:
        Dictionary with recall@k
    """
    encoder = SimpleEncoder(model_name, pooling=pooling, layer_idx=layer_idx, batch_size=batch_size)

    # Encode
    query_texts = [q["text"] for q in queries]
    corpus_texts = [c["text"] for c in corpus]

    query_embs = encoder.encode(query_texts).numpy()
    corpus_embs = encoder.encode(corpus_texts).numpy()

    # Build ID mappings
    qid_to_idx = {q["_id"]: i for i, q in enumerate(queries)}
    cid_to_idx = {c["_id"]: i for i, c in enumerate(corpus)}

    # Build relevance dict
    relevant_docs = {}
    for row in relevance:
        qid = row["query-id"]
        cid = row["corpus-id"]
        if qid in qid_to_idx and cid in cid_to_idx:
            relevant_docs.setdefault(qid, set()).add(cid)

    # Compute similarity
    sim_matrix = cosine_similarity(query_embs, corpus_embs)

    # Calculate Recall@k
    recalls = []
    for q in queries:
        qid = q["_id"]
        if qid not in relevant_docs:
            continue

        q_idx = qid_to_idx[qid]
        top_k_idx = np.argsort(sim_matrix[q_idx])[-top_k:][::-1]
        top_k_cids = [corpus[i]["_id"] for i in top_k_idx]

        is_hit = any(cid in relevant_docs[qid] for cid in top_k_cids)
        recalls.append(int(is_hit))

    recall = np.mean(recalls) if recalls else 0.0

    return {
        "recall@10": recall,
        "layer": layer_idx
    }


def evaluate_layer_sts(
    layer_idx: int,
    model_name: str,
    sentence_pairs: List[Tuple[str, str]],
    scores: List[float],
    pooling: str = "mean",
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate single layer on STS task using Spearman correlation.

    Args:
        layer_idx: Layer index
        model_name: Model name
        sentence_pairs: List of (sentence1, sentence2) tuples
        scores: Ground truth similarity scores
        pooling: Pooling strategy
        batch_size: Batch size

    Returns:
        Dictionary with Spearman correlation
    """
    from scipy.stats import spearmanr

    encoder = SimpleEncoder(model_name, pooling=pooling, layer_idx=layer_idx, batch_size=batch_size)

    # Encode all sentences
    all_sentences = []
    for s1, s2 in sentence_pairs:
        all_sentences.extend([s1, s2])

    embeddings = encoder.encode(all_sentences).numpy()

    # Compute cosine similarities
    pred_scores = []
    for i in range(0, len(embeddings), 2):
        emb1 = embeddings[i]
        emb2 = embeddings[i + 1]
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        pred_scores.append(sim)

    # Spearman correlation
    corr, pval = spearmanr(scores, pred_scores)

    return {
        "spearman": corr,
        "pvalue": pval,
        "layer": layer_idx
    }


def evaluate_all_layers(
    model_name: str,
    task_type: str,
    data: Dict,
    pooling: str = "mean",
    batch_size: int = 32
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate all layers of a model on a task.

    Args:
        model_name: HuggingFace model name
        task_type: Type of task ("classification", "retrieval", "sts")
        data: Task data dictionary
        pooling: Pooling strategy
        batch_size: Batch size

    Returns:
        Dictionary mapping layer indices to evaluation metrics
    """
    # Determine number of layers
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers + 1  # +1 for embeddings

    results = {}

    for layer_idx in range(num_layers):
        if task_type == "classification":
            metrics = evaluate_layer_classification(
                layer_idx, model_name,
                data["train_texts"], data["train_labels"],
                data["val_texts"], data["val_labels"],
                pooling, batch_size=batch_size
            )
        elif task_type == "retrieval":
            metrics = evaluate_layer_retrieval(
                layer_idx, model_name,
                data["queries"], data["corpus"], data["relevance"],
                pooling, batch_size=batch_size
            )
        elif task_type == "sts":
            metrics = evaluate_layer_sts(
                layer_idx, model_name,
                data["sentence_pairs"], data["scores"],
                pooling, batch_size=batch_size
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        results[layer_idx] = metrics

    return results


def extract_layer_quality(results: Dict[int, Dict[str, float]], metric: str = "accuracy") -> np.ndarray:
    """
    Extract quality scores from evaluation results.

    Args:
        results: Evaluation results from evaluate_all_layers
        metric: Metric name to extract (e.g., "accuracy", "recall@10", "spearman")

    Returns:
        Array of quality scores indexed by layer
    """
    num_layers = len(results)
    quality = np.zeros(num_layers)

    for layer_idx, metrics in results.items():
        quality[layer_idx] = metrics.get(metric, 0.0)

    return quality
