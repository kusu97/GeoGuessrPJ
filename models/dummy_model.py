from models.base import Prediction, BaseModel

class DummyModel(BaseModel):
    """
    A dummy baseline model that always returns the same prediction.
    """

    def __init__(self,
                 country: str = "Japan",
                 lat: float = 35.6684,
                 lng: float = 139.7077):
        """
        Default prediction: shibuya, Tokyo, Japan
        """
        self.info = {"model_name": "Dummy Model"}
        self.country = country
        self.lat = lat
        self.lng = lng

    def predict(self, image_path: str, prompt: str):
        return Prediction(
            country=self.country,
            lat=self.lat,
            lng=self.lng
        )
