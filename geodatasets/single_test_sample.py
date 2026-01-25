from geodatasets.base import BaseDataset

class SingleSample(BaseDataset):
    """
    a single test sample

    place: shibuya, Tokyo, Japan
    lat: 35.6593
    lng: 139.7006
    image_path: "./geodatasets/images_for_single_test/resized_shibuya.jpg"
    """
    def __init__(self):
        self.name = "A single test sample"
        self.image_path = "./geodatasets/images_for_single_test/resized_shibuya.jpg"
        self.country = "Japan"
        self.lat = 35.6593
        self.lng = 139.7006

    def __len__(self):
        return 1

    def get_sample(self, idx):
        if idx != 0:
            raise IndexError("There's only ONE sample in the SingleSample dataset.")
        
        return {"image_path": self.image_path, 
                "country": self.country, 
                "lat": self.lat, 
                "lng": self.lng}


if __name__ == '__main__':
    example = SingleSample()
    print(example.get_sample(0))