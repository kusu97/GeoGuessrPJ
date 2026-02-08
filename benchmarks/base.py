from abc import ABC, abstractmethod
from models.base import Prediction

class BaseBenchmark(ABC):

    @abstractmethod
    def evaluate(self, sample: dict, pred: Prediction | None) -> dict:
        """
        sample: one sample in the dataset
            MUST be a dict with keys:
            - image_path
            - lat
            - lng
            - country (optional)
        pred: model prediction (is None indicates a parser failure)
        return: dict of sample information and metrics
        """
        pass

    @abstractmethod
    def summarize(self) -> dict:
        pass

    @abstractmethod
    def save_results(self, save_dir: str, extra_meta: dict | None) -> None:
        pass