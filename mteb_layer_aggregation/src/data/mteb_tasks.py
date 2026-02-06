"""
MTEB task configuration and management.
Provides utilities for loading and organizing MTEB tasks by category.
"""
from typing import List, Dict, Optional
import mteb


MTEB_TASK_CATEGORIES = {
    "Classification": [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "ArxivClusteringP2P",
        "ArxivClusteringS2S",
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        "RedditClustering",
        "RedditClusteringP2P",
        "StackExchangeClustering",
        "StackExchangeClusteringP2P",
        "TwentyNewsgroupsClustering",
    ],
    "PairClassification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "Retrieval": [
        "ArguAna",
        "ClimateFEVER",
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
        "DBPedia",
        "FEVER",
        "FiQA2018",
        "HotpotQA",
        "MSMARCO",
        "NFCorpus",
        "NQ",
        "QuoraRetrieval",
        "SCIDOCS",
        "SciFact",
        "Touche2020",
        "TRECCOVID",
    ],
    "STS": [
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STS17",
        "STS22",
        "STSBenchmark",
    ],
    "Summarization": [
        "SummEval",
    ],
}


def get_all_task_names() -> List[str]:
    """Get all MTEB task names across all categories."""
    all_tasks = []
    for tasks in MTEB_TASK_CATEGORIES.values():
        all_tasks.extend(tasks)
    return all_tasks


def get_tasks_by_category(category: str) -> List[str]:
    """
    Get task names for a specific category.

    Args:
        category: One of "Classification", "Clustering", "PairClassification", 
                 "Reranking", "Retrieval", "STS", "Summarization"

    Returns:
        List of task names in that category
    """
    return MTEB_TASK_CATEGORIES.get(category, [])


def get_task_category(task_name: str) -> Optional[str]:
    """
    Get the category for a given task name.

    Args:
        task_name: Name of the task

    Returns:
        Category name or None if not found
    """
    for category, tasks in MTEB_TASK_CATEGORIES.items():
        if task_name in tasks:
            return category
    return None


def load_mteb_tasks(
    task_names: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    languages: List[str] = ["eng"]
) -> List:
    """
    Load MTEB tasks filtered by names or categories.

    Args:
        task_names: Specific task names to load (overrides categories)
        categories: Task categories to load (e.g., ["Classification", "Retrieval"])
        languages: Languages to filter (default: ["eng"])

    Returns:
        List of MTEB task objects
    """
    if task_names is None:
        if categories is None:
            # Load all tasks
            task_names = get_all_task_names()
        else:
            # Load tasks from specified categories
            task_names = []
            for cat in categories:
                task_names.extend(get_tasks_by_category(cat))

    # Load tasks using MTEB
    tasks = mteb.get_tasks(tasks=task_names, languages=languages)
    return tasks


class MTEBTaskConfig:
    """Configuration helper for MTEB task evaluation."""

    def __init__(
        self,
        task_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        languages: List[str] = ["eng"],
        batch_size: int = 32,
        output_folder: str = "./results"
    ):
        """
        Initialize MTEB task configuration.

        Args:
            task_names: Specific tasks to evaluate
            categories: Task categories to evaluate
            languages: Languages to include
            batch_size: Batch size for evaluation
            output_folder: Output directory for results
        """
        self.task_names = task_names
        self.categories = categories
        self.languages = languages
        self.batch_size = batch_size
        self.output_folder = output_folder

        # Load tasks
        self.tasks = load_mteb_tasks(task_names, categories, languages)

    def get_task_names(self) -> List[str]:
        """Get names of loaded tasks."""
        return [task.description["name"] for task in self.tasks]

    def get_task_count(self) -> int:
        """Get number of loaded tasks."""
        return len(self.tasks)

    def get_tasks_by_category_dict(self) -> Dict[str, List[str]]:
        """Get loaded tasks organized by category."""
        task_dict = {}
        for task in self.tasks:
            task_name = task.description["name"]
            category = get_task_category(task_name)
            if category:
                if category not in task_dict:
                    task_dict[category] = []
                task_dict[category].append(task_name)
        return task_dict
