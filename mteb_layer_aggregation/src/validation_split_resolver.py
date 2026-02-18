import mteb
from datasets import disable_caching
import datasets

class ValidationSplitResolver:
    VAL_SPLITS = ["dev", "validation", "val"]

    def __init__(self, task_name: str, verbose: int = 0):
        self.task_name = task_name
        self.verbose = verbose

        self._task = None
        self._dataset = None
        self._val_name = None
        self._task_type = None
        self._dataset_path: list[str] = []  # keys used to reach the flat dataset

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def task(self):
        self._ensure_resolved(); return self._task

    @property
    def dataset(self):
        self._ensure_resolved(); return self._dataset

    @property
    def val_name(self) -> str:
        self._ensure_resolved(); return self._val_name

    @property
    def task_type(self) -> str:
        self._ensure_resolved(); return self._task_type

    @property
    def dataset_path(self) -> list[str]:
        """Keys used to descend from the raw predictions dict to the flat split level."""
        self._ensure_resolved(); return self._dataset_path

    def unwrap_predictions(self, pred_data: dict) -> dict:
        """
        Apply the same key-path used when loading the dataset to a raw
        predictions dict, then return the per-split dict.
        """
        self._ensure_resolved()
        d = pred_data
        extended_keys = ['default'] + self._dataset_path + ['default'] 
        
        for key in extended_keys:
            if (isinstance(d, dict) or 
                isinstance(d, datasets.dataset_dict.DatasetDict)) and key in d:
                d = d[key]
        # d should now be at the same nesting level as the dataset
        if self.val_name not in d:
            raise ValueError(
                f"Split '{self.val_name}' not found. "
                f"Available: {list(d.keys())}  (path used: {self._dataset_path})"
            )
        return d[self.val_name]

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _ensure_resolved(self):
        if self._val_name is not None:
            return

        task = self._try_load_task()
        self._task_type = task.metadata.type

        # Normalize nested dataset structure, recording path
        dataset = task.dataset
        path = []
        for key in ("default", "en", "default"):
            print("type dataset", (dataset))
            if (isinstance(dataset, dict) or 
                isinstance(dataset, datasets.dataset_dict.DatasetDict)) and key in dataset:
                dataset = dataset[key]
                path.append(key)

        self._dataset_path = path  # save for unwrap_predictions()

        val_name = next(
            (s for s in self.VAL_SPLITS if s in dataset), None
        )

        if val_name is None:
            if self.verbose >= 2:
                print(f"No validation split found. Available: {list(dataset.keys())}")

            if self._task_type == "Classification":
                if self.verbose >= 2:
                    print("Creating validation split for classification")
                trainval = dataset["train"].train_test_split(test_size=0.2, seed=42)
                dataset["train"]      = trainval["train"]
                dataset["validation"] = trainval["test"]
                val_name = "validation"
            else:
                val_name = "train"

        task.__dict__["_eval_splits"] = [val_name]

        # Set n_experiments if it's a classification task
        if hasattr(task, 'n_experiments'):
            if self.verbose >= 2:
                print(f"  Setting n_experiments={1} (was {task.n_experiments})")
            task.n_experiments = 1

        if self.verbose >= 2:
            print(f"Using split: {val_name}  (path: {path})")

        self._task     = task
        self._dataset  = dataset
        self._val_name = val_name

    def _try_load_task(self):
        disable_caching()
        for split_name in self.VAL_SPLITS:
            try:
                task = mteb.get_task(
                    self.task_name,
                    languages=["eng"],
                    eval_splits=[split_name],
                    exclusive_language_filter=True,
                )
                task.load_data()
                if self.verbose >= 2:
                    print(f"Loaded split: {split_name}")
                return task
            except Exception:
                continue

        task = mteb.get_task(
            self.task_name,
            languages=["eng"],
            eval_splits=["train"],
            exclusive_language_filter=True,
        )
        task.load_data()
        return task