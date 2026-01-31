import pandas as pd
from pathlib import Path
from geodatasets.base import BaseDataset

class Osv5mDataset(BaseDataset):
    """
    Adapter for osv5m (HuggingFace osv5m dataset)

    osv5m: 
    https://github.com/gastruc/osv5m
    https://huggingface.co/datasets/osv5m/osv5m
    """

    def __init__(self, csv_path='./geodatasets/osv5m_dataset/labels/test.csv', 
                 image_dir='./geodatasets/osv5m_dataset/images/test/00', 
                 max_samples=100):
        self.name = "Osv5m test dataset 00"
        self.dataset = self._load_osv5m_csv(csv_path, image_dir, max_samples)

    def __len__(self):
        return len(self.dataset)

    def get_sample(self, idx):
        return self.dataset[idx]

    def _load_osv5m_csv(self, csv_path, image_dir, max_samples):
        
        df = pd.read_csv(csv_path)

        samples = []
        image_dir = Path(image_dir)

        for row in df.itertuples(index=False):
            image_path = image_dir / f"{row.id}.jpg"

            # Skip if the image does not exist
            if not image_path.exists():
                continue

            sample = {
                "image_path": str(image_path),
                "country": row.country,
                "lat": float(row.latitude),
                "lng": float(row.longitude),
            }

            samples.append(sample)

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples

if __name__ == '__main__':
    example = Osv5mDataset()
    print(example.get_sample(0))