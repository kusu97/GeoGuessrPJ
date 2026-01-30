from benchmarks.base import BaseBenchmark
from models.base import Prediction
from dataclasses import dataclass, asdict
import haversine
import math
import json
import time
import os

@dataclass
class GeoScoreGT:
    lat: float
    lng: float

    def to_dict(self):
        return asdict(self)

class GeoScore(BaseBenchmark):
    """
    A benchmark that calculates GeoScore using a method similar to that in the World Map in GeoGuessr.
    """

    def __init__(self):
        self.results = []       # Store the results for the entire dataset
        self.summary = {}       # Store the summary for the entire dataset
    
    def calculate_geoscore(self, distance_km):
        '''An empirical formula summarized from actual gameplay on the World map in GeoGuessr.'''
        if distance_km * 1000 <= 25:
            return 5000
        return round(5000 * math.exp(-10.0 * distance_km / 14916.862))

    def _get_GT_from_sample(self, sample: dict):
        return GeoScoreGT(lat=sample["lat"], lng=sample["lng"])

    def evaluate(self, sample: dict, pred: Prediction):
        gt = self._get_GT_from_sample(sample)
        distance = haversine.haversine(
            (gt.lat, gt.lng),
            (pred.lat, pred.lng)
        )
        geoscore = self.calculate_geoscore(distance)

        result = {
            "image_path": sample["image_path"],
            "gt": gt.to_dict(),
            "pred": pred.to_dict(),
            "metrics": {
                "distance_km": distance,
                "geoscore": geoscore,
            }
        }

        self.results.append(result)
        return result

    def summarize(self):
        if len(self.results) == 0:
            raise RuntimeError("No results to summarize. Run evaluation first.")

        self.summary =  {
            "avg_distance": sum(r["metrics"]["distance_km"] for r in self.results) / len(self.results),
            "avg_geoscore": sum(r["metrics"]["geoscore"] for r in self.results) / len(self.results),
        }

        return self.summary
    
    def save_results(self, save_dir: str = "./records/results/geoscore_results", extra_meta: dict | None = None):
        """
        Save per-sample results and summary to a JSON file.
        (self.summarize() MUST be run before saving the results)
        """
        if len(self.results) == 0:
            raise RuntimeError("No results to save. Run evaluation first.")

        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.json"
        save_path = os.path.join(save_dir, filename)

        output = {
            "meta": {
                "benchmark": "GeoScore",
                "num_samples": len(self.results),
                "timestamp": timestamp,
            },
            "summary": self.summary,
            "results": self.results,
        }

        if extra_meta is not None:
            output["meta"].update(extra_meta)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    pass