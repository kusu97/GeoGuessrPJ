from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

@dataclass
class Prediction:
    country: str
    lat: float
    lng: float

    def to_dict(self):
        return asdict(self)

class BaseModel(ABC):

    @abstractmethod
    def predict(self, image_path: str, prompt: str) -> Prediction | None:
        """
        image_path: path to image
        prompt: full prompt text
        return: structured prediction (country, lat, lng)
        """
        pass