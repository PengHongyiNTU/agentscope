"""The PersonaMem benchmark class in agentscope. The code is implemented with
reference to the `PersonaMem <https://github.com/bowen-upenn/PersonaMem/>`
"""


from typing import Generator, Literal
from .._benchmark_base import BenchmarkBase
from .._task import Task
from ._personamem_metric import PersonaMemMCQAccuracy
import os
import csv
import huggingface_hub


class PersonaMemBenchmark(BenchmarkBase):
    """
    The PersonaMem benchmark for evaluating how well language models can infer
    evolving user profiles and generate personalized responses across
    task scenarios.

    Paper: https://arxiv.org/abs/2504.14225
    GitHub: https://github.com/bowen-upenn/PersonaMem
    Hugging Face Repo: https://huggingface.co/datasets/bowen-upenn/PersonaMem
    """

    huggingface_repo_id: str = "bowen-upenn/PersonaMem"

    def __init__(
        self,
        data_dir: str,
        split: Literal["32k", "128k", "1M"] = "32k",
    ) -> None:
        """Initialize the PersonaMem benchmark.

        Args:
            data_dir (`str`):
                The directory where the PersonaMem dataset is downloaded and
                stored.
            split (`Literal["32k", "128k", "1M"]`, *optional*, defaults to
            "32k"):
                The split of the PersonaMem dataset to use.
        """
        super().__init__(
            name="PersonaMem Benchmark",
            description=(
                "The PersonaMem benchmark for evaluating how well language "
                "models can infer evolving user profiles and generate "
                "personalized responses across task scenarios."
            ),
        )

        self.data_dir = os.path.abspath(data_dir)
        self.split = split

        if os.path.exists(self.data_dir) and not os.path.isdir(self.data_dir):
            raise RuntimeError(
                f"The data_dir '{data_dir} is not valid directory path.'"
            )

        os.makedirs(self.data_dir, exist_ok=True)

        if not self._verify_data():
            self._download_data()

        self.dataset = self._load_data()

    def _verify_data(self) -> bool:
        """Verify if the required data files exist in the data directory."""
        required_files = [
            f"shared_contexts_{self.split}.jsonl",
            f"questions_{self.split}.csv",
        ]
        for file_name in required_files:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def _download_data(self) -> None:
        """Download the required data files from HuggingFace"""

        files_to_download = [
            f"shared_contexts_{self.split}.jsonl",
            f"questions_{self.split}.csv",
        ]
        for file_name in files_to_download:
            huggingface_hub.hf_hub_download(
                repo_id=self.huggingface_repo_id,
                filename=file_name,
                repo_type="dataset",
                local_dir=self.data_dir,
            )

    def _load_data(self) -> list[dict]:
        # csv file are relatively small, we can load them into memory
        dataset = []
        csv_file_path = os.path.join(
            self.data_dir,
            f"questions_{self.split}.csv",
        )
        with open(csv_file_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                dataset.append(row)
        return dataset

    @staticmethod
    def _data_to_task(data: dict) -> Task:
        """Convert a data record to a Task object."""
        question_id = data.pop("question_id")
        input = (
            f"All options: {data.pop('all_options')}\n"
            f"Question: {data.pop('user_question_or_message')}"
        )
        ground_truth = data.pop("correct_answer")
        tags = {
            "persona_id": data.pop("persona_id", ""),
            "question_type": data.pop("question_type", ""),
            "topic": data.pop("topic", ""),
        }
        metadata = data
        task = Task(
            id=question_id,
            input=input,
            ground_truth=ground_truth,
            tags=tags,
            metrics=[PersonaMemMCQAccuracy(ground_truth)],
            metadata=metadata,
        )
        return task

    def __getitem__(self, index: int) -> Task:
        """Get a task by index."""
        data = self.dataset[index]
        return self._data_to_task(data)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset)

    def __iter__(self) -> Generator[Task, None, None]:
        """Create a generator to iterate over the tasks in the dataset."""
        for data in self.dataset:
            yield self._data_to_task(data)
