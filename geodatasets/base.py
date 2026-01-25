from abc import ABC, abstractmethod

class BaseDataset(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_sample(self, idx):
        """
        return: dict with keys:
        - image_path
        - country
        - lat
        - lng
        """
        pass
