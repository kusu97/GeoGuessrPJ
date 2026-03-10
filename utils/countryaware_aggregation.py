'''
A script performing country-aware aggregation for multiple geolocation predictions, 
used in the self-consistency experiment.
'''

from models.base import Prediction
from utils.canon import standardize_country_name
import haversine
from collections import Counter


def compute_medoid(points):
    best_point = None
    best_score = float("inf")

    for i, p in enumerate(points):

        total_dist = 0

        for j, q in enumerate(points):
            if i != j:
                total_dist += haversine.haversine(p, q)

        if total_dist < best_score:
            best_score = total_dist
            best_point = p

    return best_point


def countryaware_aggregation(predictions: list[Prediction]) -> Prediction:
    '''
    This function aggregates multiple predictions produced by a multimodal large
    language model (MLLM) using a country-aware voting strategy combined with a
    medoid-based coordinate selection.

    The aggregation procedure consists of the following steps:

    1. Obtain K model predictions in the format (country, latitude, longitude).
    2. Perform majority voting over predicted countries.
    3. Filter predictions that belong to the majority country.
    4. Compute the medoid of the remaining coordinates using the Haversine distance.
        The medoid is defined as the point whose total geodesic distance to all other
        points in the set is minimal.
    5. Return the final aggregated prediction in the same format.

    This strategy improves robustness by:
    - Reducing the influence of outlier predictions.
    - Preventing aggregation across different countries.
    - Avoiding centroid locations that may fall outside the predicted country
        (e.g., oceans or neighboring countries).
    '''

    # Step 1: country vote
    countries = [standardize_country_name(pred.country) for pred in predictions]
    country_counter = Counter(countries)

    majority_country = country_counter.most_common(1)[0][0]

    # Step 2: only retain majority country
    filtered = [
        pred for pred in predictions
        if standardize_country_name(pred.country) == majority_country
    ]

    # Step 3: extract the (lat, lng)
    coords = [(pred.lat, pred.lng) for pred in filtered]

    # Step 4: medoid
    final_lat, final_lng = compute_medoid(coords)

    return Prediction(country=majority_country, lat=final_lat, lng=final_lng)


if __name__ == "__main__":
    pass