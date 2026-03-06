'''
Perform a per-sample comparison between two experimental results and 
identify the samples with large prediction differences.

Note: 
1. The dataset must be the same, and the benchmark must be OSV5MAdapter.
2. If an image fails in prediction in either of the experiments, it will not be included.
'''

import json
import haversine

def find_top_diff_samples(results1_path, results2_path, pred_threshold=1000, 
                          err_threshold=1000, score_threshold=1000, top_k=5):

    with open(results1_path) as f:
        run1 = json.load(f)

    with open(results2_path) as f:
        run2 = json.load(f)
    
    results1 = {r["image_path"]: r for r in run1["results"]}
    results2 = {r["image_path"]: r for r in run2["results"]}
    
    diffs = []
    for img in results1:
        a = results1[img]
        
        if img not in results2:
            continue
        b = results2[img]

        latA = a["pred"]["lat"]
        lonA = a["pred"]["lng"]

        latB = b["pred"]["lat"]
        lonB = b["pred"]["lng"]

        pred_diff = haversine.haversine((latA, lonA), (latB, lonB))

        distA = a["metrics"]["distance_km"]
        distB = b["metrics"]["distance_km"]

        err_diff = abs(distA - distB)

        scoreA = a["metrics"]["geoscore"]
        scoreB = b["metrics"]["geoscore"]

        score_diff = abs(scoreA - scoreB)

        if pred_diff > pred_threshold and err_diff > err_threshold and score_diff > score_threshold:
            diffs.append({
                "image_path": img,
                "pred_distance": pred_diff,
                "err_diff": err_diff,
                "score_diff": score_diff,
                "result1": a,
                "result2": b
            })

    diffs.sort(key=lambda x: x["pred_distance"], reverse=True)

    print("Large difference samples:", len(diffs))

    print(f"\nTop {top_k} different samples: ")
    for diff in diffs[:top_k]:
        print(json.dumps(diff, indent=4, ensure_ascii=False), end="\n\n")

    return diffs[:top_k]

if __name__ == "__main__":
    result1_path = "your result1 path"
    result2_path = "your result2 path"

    top_diff_samples = find_top_diff_samples(result1_path, result2_path)