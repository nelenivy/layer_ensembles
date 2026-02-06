"""
MTEB evaluation runner for aggregated encoders.
"""
import mteb
import json
from pathlib import Path
from typing import Dict, List, Optional
from ..models.encoders import BaseEncoder


def run_mteb_evaluation(
    encoder: BaseEncoder,
    tasks: List,
    output_folder: str,
    batch_size: Optional[int] = None,
    verbosity: int = 1
) -> List[Dict]:
    """
    Run MTEB evaluation on encoder.

    Args:
        encoder: Encoder to evaluate
        tasks: List of MTEB tasks
        output_folder: Output directory
        batch_size: Override encoder batch size
        verbosity: Logging verbosity

    Returns:
        List of evaluation results
    """
    if batch_size is not None:
        encoder.batch_size = batch_size

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        encoder,
        output_folder=output_folder,
        verbosity=verbosity
    )

    return results


def extract_main_scores(results: List[Dict], split: str = "test") -> Dict[str, float]:
    """
    Extract main scores from MTEB results.

    Args:
        results: MTEB evaluation results
        split: Data split ("test", "validation", "dev")

    Returns:
        Dictionary mapping task names to main scores
    """
    scores = {}
    for result in results:
        task_name = result.get("task_name", "Unknown")

        # Navigate to scores
        if "scores" in result:
            split_scores = result["scores"].get(split, {})
            if isinstance(split_scores, list) and len(split_scores) > 0:
                split_scores = split_scores[0]

            # Get main score
            main_score = split_scores.get("main_score")
            if main_score is not None:
                scores[task_name] = main_score

    return scores


def save_mteb_results(
    results: List[Dict],
    output_path: str
):
    """
    Save MTEB results to JSON file.

    Args:
        results: MTEB evaluation results
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_mteb_results(result_path: str) -> Dict:
    """
    Load MTEB results from JSON file.

    Args:
        result_path: Path to results JSON

    Returns:
        Dictionary of results
    """
    with open(result_path, 'r') as f:
        return json.load(f)


def aggregate_results_by_category(
    results: List[Dict],
    split: str = "test"
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results by task category.

    Args:
        results: MTEB evaluation results
        split: Data split

    Returns:
        Dictionary mapping categories to aggregated metrics
    """
    from ..data.mteb_tasks import get_task_category

    category_scores = {}

    for result in results:
        task_name = result.get("task_name", "Unknown")
        category = get_task_category(task_name)

        if category is None:
            continue

        # Get main score
        if "scores" in result:
            split_scores = result["scores"].get(split, {})
            if isinstance(split_scores, list) and len(split_scores) > 0:
                split_scores = split_scores[0]

            main_score = split_scores.get("main_score")
            if main_score is not None:
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(main_score)

    # Compute averages
    aggregated = {}
    for category, scores in category_scores.items():
        aggregated[category] = {
            "mean": sum(scores) / len(scores),
            "count": len(scores),
            "scores": scores
        }

    return aggregated
