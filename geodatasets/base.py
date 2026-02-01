from abc import ABC, abstractmethod

class BaseDataset(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_sample(self, idx):
        """
        return: dict
            Must contain the following keys:
            - image_path
            - lat
            - lng
            May contain additional keys:
            - country
        """
        pass
