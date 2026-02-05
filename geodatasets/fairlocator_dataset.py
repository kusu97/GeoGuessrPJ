import pandas as pd
from pathlib import Path
from geodatasets.base import BaseDataset

class FairLocatorDataset(BaseDataset):
    """
    Adapter for FairLocator dataset

    Description:
        FairLocator dataset consists of two subsets: "Breath" and "Depth", each containing 600 images.

    Reference:
        github: https://github.com/limenlp/FairLocator
        Paper: https://aclanthology.org/2025.emnlp-main.910/
        (Jingyuan Huang, Jen-tse Huang, Ziyi Liu, Xiaoyuan Liu, Wenxuan Wang, and Jieyu Zhao. 2025. 
        AI Sees Your Location—But With A Bias Toward The Wealthy World. 
        In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 
        pages 18019–18039, Suzhou, China. Association for Computational Linguistics.)
    """

    def __init__(self, dataset_name, max_samples=None):
        
        self.name = f"FairLocator {dataset_name} dataset"
        
        if dataset_name in ("Breadth", "Depth"):
            xlsx_path = f"./geodatasets/fairlocator_dataset/labels/{dataset_name}.xlsx"
            image_dir = f"./geodatasets/fairlocator_dataset/images/{dataset_name}"
        else:
            raise ValueError('dataset_name MUST be "Breadth" or "Depth"')

        self.dataset = self._load_dataset(xlsx_path, image_dir, max_samples)

    def __len__(self):
        return len(self.dataset)

    def get_sample(self, idx):
        return self.dataset[idx]

    def _load_dataset(self, xlsx_path, image_dir, max_samples):
        
        df = pd.read_excel(xlsx_path)

        samples = []
        image_dir = Path(image_dir)

        for _, row in df.iterrows():
            image_path = image_dir / f"{row["ID"]}.jpg"

            latlng = row["DataPoint"]
            lat, lng = map(float, latlng.split(","))

            sample = {
                "image_path": str(image_path),
                "continent": row["continent"],
                "country": row["country"],
                "city": row["city"],
                "street": row["street"],
                "lat": lat,
                "lng": lng,
            }

            samples.append(sample)

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples

if __name__ == '__main__':
    example = FairLocatorDataset("Breadth")
    print(example.get_sample(0))