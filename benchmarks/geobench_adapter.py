from benchmarks.base import BaseBenchmark
from utils.canon import are_same_country
from models.base import Prediction
from dataclasses import dataclass, asdict
import haversine
import math
import json
import time
import os

@dataclass
class GeoBenchGT:
    country: str
    lat: float
    lng: float

    def to_dict(self):
        return asdict(self)

class GeoBenchAdapter(BaseBenchmark):
    """
    Adapter for GeoBench-style evaluation
    """

    def __init__(self):
        self.scale = self._compute_scale()
        self.results = []       # Store the results for the entire dataset
        self.summary = {}       # Store the summary for the entire dataset

    def _compute_scale(self):
        '''Fix the scale parameter to a global constant corresponding to Earth-scale GeoGuessr scoring'''
        EARTH_RADIUS_KM = 6371.0
        GLOBAL_DIAMETER = math.pi * EARTH_RADIUS_KM
        return GLOBAL_DIAMETER / 7.458421       # 2683.555513428455
    
    def _calculate_score(self, distance_km):
        if distance_km * 1000000 <= 25:
            return 5000
        return round(5000 * math.pow(0.99866017, (distance_km * 1000000) / self.scale))

    def _get_GT_from_sample(self, sample: dict):
        return GeoBenchGT(country=sample["country"], lat=sample["lat"], lng=sample["lng"])

    def evaluate(self, sample: dict, pred: Prediction):
        gt = self._get_GT_from_sample(sample)
        distance = haversine.haversine(
            (gt.lat, gt.lng),
            (pred.lat, pred.lng)
        )
        score = self._calculate_score(distance / 1000)
        country_correct = are_same_country(gt.country, pred.country)

        result = {
            "image_path": sample["image_path"],
            "gt": gt.to_dict(),
            "pred": pred.to_dict(),
            "metrics": {
                "distance_km": distance,
                "score": score,
                "country_correct": country_correct,
            }
        }

        self.results.append(result)
        return result

    def summarize(self):
        if len(self.results) == 0:
            raise RuntimeError("No results to summarize. Run evaluation first.")

        self.summary =  {
            "avg_distance": sum(r["metrics"]["distance_km"] for r in self.results) / len(self.results),
            "avg_score": sum(r["metrics"]["score"] for r in self.results) / len(self.results),
            "country_acc": sum(r["metrics"]["country_correct"] for r in self.results) / len(self.results)
        }

        return self.summary
    
    def save_results(self, save_dir: str = "./records/results/geobench_results", extra_meta: dict | None = None):
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
                "benchmark": "GeoBench",
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