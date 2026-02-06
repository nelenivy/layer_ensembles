"""
Train PCA transformations for layer aggregation.
Matches notebook PCA training procedures.
"""
import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pickle
import os
import gc
from typing import List, Optional


def train_pca_for_selected_layers(
    model_name: str,
    layer_indices: List[int],
    layer_weights: np.ndarray,
    num_samples: int = 30000,
    output_dim: int = 768,
    batch_size: int = 128,
    pooling: str = "mean",
    dataset_name: str = "allenai/c4",
    dataset_subset: str = "en",
    output_path: str = "./pca_qp.pkl",
    device: str = "cuda"
):
    """
    Train PCA for QP-weighted layer concatenation (concat_pca_qp).

    Args:
        model_name: HuggingFace model name
        layer_indices: Which layers to use
        layer_weights: Weights for each layer
        num_samples: Number of C4 samples to use
        output_dim: PCA output dimension
        batch_size: Batch size for encoding
        pooling: "mean" or "cls"
        dataset_name: Dataset to use
        dataset_subset: Dataset subset
        output_path: Where to save PCA
        device: Device to use

    Returns:
        Dict with PCA components, mean, and metadata
    """
    print(f"Training PCA for selected layers: {layer_indices}")
    print(f"Output dim: {output_dim}, Samples: {num_samples}")

    # Load dataset
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)
    texts = [ex["text"] for ex in ds.shuffle(buffer_size=num_samples, seed=42).take(num_samples)]
    print(f"✓ Loaded {len(texts)} samples")

    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    print(f"✓ Loaded model: {model_name}")

    # Prepare weights
    weights_sqrt = torch.tensor(layer_weights, dtype=torch.float32).sqrt()
    concat_dim = len(layer_indices) * 768  # Assuming 768 hidden dim

    # Initialize PCA
    ipca = IncrementalPCA(n_components=output_dim, batch_size=512)
    buffer = []
    buffer_size = 0
    collect_threshold = max(output_dim, 2048)

    def flush_to_pca():
        nonlocal buffer, buffer_size
        if buffer_size > 0:
            ipca.partial_fit(np.vstack(buffer))
            buffer.clear()
            buffer_size = 0
            torch.cuda.empty_cache()
            gc.collect()

    # Extract and concatenate weighted layers
    print("Extracting embeddings...")
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="PCA training"):
            batch = texts[start:start + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # Extract weighted layers
            weighted_layers = []
            for idx, layer_idx in enumerate(layer_indices):
                layer_hidden = hidden_states[layer_idx]  # (B, T, H)

                # Pool
                if pooling == "cls":
                    pooled = layer_hidden[:, 0, :]
                else:  # mean
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (layer_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

                # Apply sqrt(weight)
                weighted = pooled * weights_sqrt[idx].to(pooled.device)
                weighted_layers.append(weighted)

            # Concatenate
            concat = torch.cat(weighted_layers, dim=-1).cpu().numpy()  # (B, concat_dim)

            buffer.append(concat)
            buffer_size += concat.shape[0]

            if buffer_size >= collect_threshold:
                flush_to_pca()

            del outputs, hidden_states, inputs, concat
            torch.cuda.empty_cache()

    # Final flush
    flush_to_pca()

    print(f"✓ PCA trained: {concat_dim} → {output_dim}")

    # Save PCA
    result = {
        "components": ipca.components_,  # (output_dim, concat_dim)
        "mean": ipca.mean_,  # (concat_dim,)
        "explained_variance": ipca.explained_variance_,
        "layer_indices": layer_indices,
        "layer_weights": layer_weights,
        "model_name": model_name,
        "pooling": pooling,
        "num_samples": num_samples,
        "output_dim": output_dim,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"✓ Saved PCA: {output_path}")
    print(f"  Explained variance ratio: {ipca.explained_variance_ratio_.sum():.4f}")

    return result


def train_pca_for_clusters(
    model_name: str,
    clusters: List[List[int]],
    cluster_weights: np.ndarray,
    num_samples: int = 30000,
    output_dim: int = 768,
    batch_size: int = 128,
    pooling: str = "mean",
    dataset_name: str = "allenai/c4",
    dataset_subset: str = "en",
    output_path: str = "./pca_cluster.pkl",
    device: str = "cuda"
):
    """
    Train PCA for hierarchical cluster concatenation (concat_pca_cluster).

    Args:
        model_name: HuggingFace model name
        clusters: List of layer clusters
        cluster_weights: Weight for each cluster
        num_samples: Number of samples
        output_dim: PCA output dimension
        batch_size: Batch size
        pooling: "mean" or "cls"
        dataset_name: Dataset to use
        dataset_subset: Dataset subset
        output_path: Where to save
        device: Device

    Returns:
        Dict with PCA and metadata
    """
    print(f"Training PCA for {len(clusters)} clusters")
    print(f"Clusters: {clusters}")

    # Load dataset
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)
    texts = [ex["text"] for ex in ds.shuffle(buffer_size=num_samples, seed=42).take(num_samples)]
    print(f"✓ Loaded {len(texts)} samples")

    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    print(f"✓ Loaded model: {model_name}")

    # Prepare weights
    weights_sqrt = torch.tensor(cluster_weights, dtype=torch.float32).sqrt()
    concat_dim = len(clusters) * 768

    # Initialize PCA
    ipca = IncrementalPCA(n_components=output_dim, batch_size=512)
    buffer = []
    buffer_size = 0
    collect_threshold = max(output_dim, 2048)

    def flush_to_pca():
        nonlocal buffer, buffer_size
        if buffer_size > 0:
            ipca.partial_fit(np.vstack(buffer))
            buffer.clear()
            buffer_size = 0
            torch.cuda.empty_cache()
            gc.collect()

    # Extract cluster representations
    print("Extracting cluster embeddings...")
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="PCA training"):
            batch = texts[start:start + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # Process each cluster
            cluster_vecs = []
            for cluster_idx, cluster_layers in enumerate(clusters):
                # Average layers within cluster
                layer_embeddings = []
                for layer_idx in cluster_layers:
                    layer_hidden = hidden_states[layer_idx]

                    # Pool
                    if pooling == "cls":
                        pooled = layer_hidden[:, 0, :]
                    else:  # mean
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        pooled = (layer_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

                    layer_embeddings.append(pooled)

                # Average within cluster
                cluster_avg = torch.stack(layer_embeddings, dim=0).mean(0)

                # Apply sqrt(weight)
                weighted = cluster_avg * weights_sqrt[cluster_idx].to(cluster_avg.device)
                cluster_vecs.append(weighted)

            # Concatenate clusters
            concat = torch.cat(cluster_vecs, dim=-1).cpu().numpy()

            buffer.append(concat)
            buffer_size += concat.shape[0]

            if buffer_size >= collect_threshold:
                flush_to_pca()

            del outputs, hidden_states, inputs, concat
            torch.cuda.empty_cache()

    # Final flush
    flush_to_pca()

    print(f"✓ PCA trained: {concat_dim} → {output_dim}")

    # Save
    result = {
        "components": ipca.components_,
        "mean": ipca.mean_,
        "explained_variance": ipca.explained_variance_,
        "clusters": clusters,
        "cluster_weights": cluster_weights,
        "model_name": model_name,
        "pooling": pooling,
        "num_samples": num_samples,
        "output_dim": output_dim,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"✓ Saved PCA: {output_path}")
    print(f"  Explained variance ratio: {ipca.explained_variance_ratio_.sum():.4f}")

    return result


def train_pca_all_layers(
    model_name: str,
    num_samples: int = 30000,
    output_dim: int = 768,
    batch_size: int = 128,
    pooling: str = "mean",
    dataset_name: str = "allenai/c4",
    dataset_subset: str = "en",
    output_path: str = "./pca_all.pkl",
    device: str = "cuda"
):
    """
    Train PCA for all layers concatenation (baseline).

    Args:
        model_name: HuggingFace model name
        num_samples: Number of samples
        output_dim: PCA output dimension
        batch_size: Batch size
        pooling: "mean" or "cls"
        dataset_name: Dataset to use
        dataset_subset: Dataset subset
        output_path: Where to save
        device: Device

    Returns:
        Dict with PCA and metadata
    """
    print(f"Training PCA for all layers (baseline)")

    # Load dataset
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)
    texts = [ex["text"] for ex in ds.shuffle(buffer_size=num_samples, seed=42).take(num_samples)]
    print(f"✓ Loaded {len(texts)} samples")

    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    num_layers = model.config.num_hidden_layers + 1
    print(f"✓ Loaded model: {model_name} ({num_layers} layers)")

    concat_dim = num_layers * 768

    # Initialize PCA
    ipca = IncrementalPCA(n_components=output_dim, batch_size=512)
    buffer = []
    buffer_size = 0
    collect_threshold = max(output_dim, 2048)

    def flush_to_pca():
        nonlocal buffer, buffer_size
        if buffer_size > 0:
            ipca.partial_fit(np.vstack(buffer))
            buffer.clear()
            buffer_size = 0
            torch.cuda.empty_cache()
            gc.collect()

    # Extract all layers
    print("Extracting all layer embeddings...")
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="PCA training"):
            batch = texts[start:start + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # Pool all layers
            pooled_layers = []
            for layer_hidden in hidden_states:
                if pooling == "cls":
                    pooled = layer_hidden[:, 0, :]
                else:  # mean
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (layer_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

                pooled_layers.append(pooled)

            # Concatenate all
            concat = torch.cat(pooled_layers, dim=-1).cpu().numpy()

            buffer.append(concat)
            buffer_size += concat.shape[0]

            if buffer_size >= collect_threshold:
                flush_to_pca()

            del outputs, hidden_states, inputs, concat
            torch.cuda.empty_cache()

    # Final flush
    flush_to_pca()

    print(f"✓ PCA trained: {concat_dim} → {output_dim}")

    # Save
    result = {
        "components": ipca.components_,
        "mean": ipca.mean_,
        "explained_variance": ipca.explained_variance_,
        "num_layers": num_layers,
        "model_name": model_name,
        "pooling": pooling,
        "num_samples": num_samples,
        "output_dim": output_dim,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"✓ Saved PCA: {output_path}")
    print(f"  Explained variance ratio: {ipca.explained_variance_ratio_.sum():.4f}")

    return result


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train PCA for layer aggregation")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, 
                        choices=["selected", "cluster", "all"])
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--output-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])

    # For selected layers
    parser.add_argument("--layer-indices", type=int, nargs="+")
    parser.add_argument("--layer-weights", type=float, nargs="+")

    # For clusters
    parser.add_argument("--clusters", type=str, help="JSON string of clusters")
    parser.add_argument("--cluster-weights", type=float, nargs="+")

    args = parser.parse_args()

    if args.method == "selected":
        train_pca_for_selected_layers(
            model_name=args.model_name,
            layer_indices=args.layer_indices,
            layer_weights=np.array(args.layer_weights),
            num_samples=args.num_samples,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            pooling=args.pooling,
            output_path=args.output_path
        )
    elif args.method == "cluster":
        import json
        clusters = json.loads(args.clusters)
        train_pca_for_clusters(
            model_name=args.model_name,
            clusters=clusters,
            cluster_weights=np.array(args.cluster_weights),
            num_samples=args.num_samples,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            pooling=args.pooling,
            output_path=args.output_path
        )
    elif args.method == "all":
        train_pca_all_layers(
            model_name=args.model_name,
            num_samples=args.num_samples,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            pooling=args.pooling,
            output_path=args.output_path
        )
