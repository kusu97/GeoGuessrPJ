from benchmarks.base import BaseBenchmark
from models.base import Prediction
from dataclasses import dataclass, asdict
import haversine
import math
import json
import time
import os
import reverse_geocoder as rg

@dataclass
class OSV5MGT:
    lat: float
    lng: float

    def to_dict(self):
        return asdict(self)

class OSV5MAdapter(BaseBenchmark):
    """
    Adapter for osv5m-style evaluation
    
    The accuracy of predicted location across geolocation models is measured 
    with three complementary sets of metrics:
        - Haversine distance d, between predicted and ground truth image locations;
        - Geoscore, based on the famous GeoGuessr game, defined as 5000 exp(-d/1492.7);
        - Accuracy of predicted locations across administrative boundaries: country, region, area, and city.

    Metrics:
    - Haversine Distance
    - GeoGuessr Score (Geoscore)
    - Radius Accuracy
    - Administrative Accuracy (country/region/area/city)    
    
    References:
    osv5m: https://github.com/gastruc/osv5m
    paper: Astruc, G., Dufour, N., Siglidis, I., Aronssohn, C., Bouia, N., Fu, S., Loiseau, R., Nguyen, V. N., 
            Raude, C., Vincent, E., XU, L., Zhou, H., & Landrieu, L. (2024). 
            OpenStreetView-5M: The Many Roads to Global Visual Geolocation (Version 1). 
            arXiv. https://doi.org/10.48550/ARXIV.2404.18873

    Note:
    To ensure compatibility with other datasets, this benchmark uniformly uses the reverse_geocoder library to 
        generate administrative labels from the latitude and longitude of each sample (and likewise for the prediction).
    The administrative labels provided in the native OSV-5M dataset are not directly imported.
    In fact, the original OSV-5M paper states that the dataset itself generates administrative labels 
        using the reverse_geocoder library, so this approach is consistent with the official methodology.
    """

    def __init__(self, radius_list=(5, 50, 200, 1000, 2500)):

        self.radius_list = radius_list      # km
        self.area_list = ("country", "region", "area", "city")

        self.results = []       # Store the results for the entire dataset
        self.summary = {}       # Store the summary for the entire dataset

    def _get_GT_from_sample(self, sample: dict):
        return OSV5MGT(
            lat=sample["lat"],
            lng=sample["lng"]
        )

    def calculate_geoscore(self, distance_km):
        return round(5000 * math.exp(-distance_km / 1492.7))

    def reverse_geocode(self, lat, lng):
        location = rg.search((lat, lng), mode=1, verbose=False)[0]

        return {
            "country": location.get('cc'),
            "region": location.get('admin1'),
            "area": location.get('admin2'), 
            "city": location.get('name')
        }

    def evaluate(self, sample: dict, pred: Prediction):
        gt = self._get_GT_from_sample(sample)
        distance = haversine.haversine(
            (gt.lat, gt.lng),
            (pred.lat, pred.lng)
        )
        geoscore = self.calculate_geoscore(distance)

        # radius accuracy
        radius_hits = {f"acc@{r}km": distance <= r for r in self.radius_list}

        # administrative accuracy
        gt_area = self.reverse_geocode(gt.lat, gt.lng)
        pred_area = self.reverse_geocode(pred.lat, pred.lng)

        area_hits = {}
        already_false = False
        for area in self.area_list:
            gt_val = gt_area.get(area)
            pred_val = pred_area.get(area)

            if gt_val is None or gt_val == "":
                if pred_val is not None or pred_val != "":
                    already_false = True
                continue
            
            if already_false:
                area_hits[f"{area}_correct"] = False
            elif pred_val == gt_val:
                area_hits[f"{area}_correct"] = True
            else:
                area_hits[f"{area}_correct"] = False
                already_false = True

        result = {
            "image_path": sample["image_path"],
            "gt": gt.to_dict(),
            "pred": pred.to_dict(),
            "gt_administrative_labels": gt_area, 
            "pred_administrative_labels": pred_area, 
            "metrics": {
                "distance_km": distance,
                "geoscore": geoscore,
                **radius_hits,
                **area_hits
            }
        }

        self.results.append(result)
        return result

    def summarize(self):
        if len(self.results) == 0:
            raise RuntimeError("No results to summarize. Run evaluation first.")

        summary =  {
            "avg_distance": sum(r["metrics"]["distance_km"] for r in self.results) / len(self.results),
            "avg_score": sum(r["metrics"]["geoscore"] for r in self.results) / len(self.results),
        }

        # radius aggregation
        for r in self.radius_list:
            key = f"acc@{r}km"
            summary[key] = sum(result["metrics"][key] for result in self.results) / len(self.results)

        # area aggregation
        for area in self.area_list:
            key = f"{area}_correct"

            valid = [
                result["metrics"].get(key)
                for result in self.results
                if key in result["metrics"]
            ]

            count = len(valid)
            summary[f"{area}_valid"] = count

            if count > 0:
                summary[f"{area}_acc"] = sum(valid) / count

        self.summary = summary
        return self.summary
    
    def save_results(self, save_dir: str = "./records/results/osv5m_results", extra_meta: dict | None = None):
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
                "benchmark": "OSV5M",
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