from abc import ABC, abstractmethod
from models.base import Prediction

class BaseBenchmark(ABC):

    @abstractmethod
    def evaluate(self, sample: dict, pred: Prediction) -> dict:
        """
        sample: one sample in the dataset
            MUST be a dict with keys:
            - image_path
            - country
            - lat
            - lng
        pred: model prediction
        return: dict of sample information and metrics
        """
        pass

    @abstractmethod
    def summarize(self):
        pass
