"""
Pooling strategies for sentence embeddings.

Supports: mean, CLS, max, last_token, pooler (trained pooler output).
"""

import torch
from typing import Dict, Callable


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling with attention mask.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]

    Returns:
        pooled: [batch, hidden_dim]
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    CLS token pooling (first token).

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len] (unused)

    Returns:
        pooled: [batch, hidden_dim]
    """
    return hidden_states[:, 0, :]


def max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Max pooling over sequence dimension.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]

    Returns:
        pooled: [batch, hidden_dim]
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    hidden_states_masked = hidden_states.clone()
    hidden_states_masked[mask_expanded == 0] = -1e9
    return torch.max(hidden_states_masked, dim=1)[0]


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Last token pooling (last non-padding token).

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]

    Returns:
        pooled: [batch, hidden_dim]
    """
    batch_size = hidden_states.shape[0]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    last_hidden = hidden_states[torch.arange(batch_size), sequence_lengths]
    return last_hidden


def pooler_output_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor, 
                          pooler_output: torch.Tensor = None) -> torch.Tensor:
    """
    Use model's trained pooler output (e.g., BERT's pooler).

    Args:
        hidden_states: [batch, seq_len, hidden_dim] (unused)
        attention_mask: [batch, seq_len] (unused)
        pooler_output: [batch, hidden_dim] (required)

    Returns:
        pooled: [batch, hidden_dim]
    """
    if pooler_output is None:
        raise ValueError("pooler_output is required for pooler strategy")
    return pooler_output


# Registry of available pooling functions
POOLING_STRATEGIES: Dict[str, Callable] = {
    'mean': mean_pooling,
    'cls': cls_pooling,
    'max': max_pooling,
    'last_token': last_token_pooling,
    'pooler': pooler_output_pooling,
}


def get_pooling_function(strategy: str) -> Callable:
    """
    Get pooling function by name.

    Args:
        strategy: Name of pooling strategy

    Returns:
        Pooling function

    Raises:
        ValueError: If strategy not found
    """
    if strategy not in POOLING_STRATEGIES:
        raise ValueError(
            f"Unknown pooling strategy: {strategy}. "
            f"Available: {list(POOLING_STRATEGIES.keys())}"
        )
    return POOLING_STRATEGIES[strategy]
