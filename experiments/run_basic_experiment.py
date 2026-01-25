'''
This script runs a complete end-to-end baseline experiment for the GeoGuessr task.

The experiment covers the full evaluation pipeline, including:
1. Loading a geolocation dataset consisting of images and ground-truth geographic annotations.
2. Feeding images together with a specified prompt into a multimodal large language model (MLLM)
to obtain structured geographic predictions (country, latitude, longitude).
3. Evaluating the model predictions using a benchmark module to compute quantitative metrics
such as geographic distance, GeoGuessr-style scores, and country-level accuracy.

This script is intended to serve as a minimal, reproducible baseline experiment.
All extended experiments (e.g., prompt ablation and model comparison)
are built upon the same pipeline structure defined here.
'''

from geodatasets.single_test_sample import SingleSample
from geodatasets.osv5m_dataset import Osv5mDataset
from benchmarks.geobench_adapter import GeoBenchAdapter
from models.dummy_model import DummyModel
from models.qwen import QwenModel

def main():
    # 1. Prepare the dataset
    # dataset = SingleSample()
    # dataset = Osv5mDataset(max_samples=10)
    dataset = Osv5mDataset(max_samples=1)

    # 2. Load the prompt
    prompt_path = "./prompts/base.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    # 3. Initialize the benchmark
    benchmark = GeoBenchAdapter()

    # 4. Initialize the model
    # model = DummyModel()
    # model = QwenModel(model_name="qwen-vl-max-2025-04-08", enable_thinking=False, save_responses=True)
    model = QwenModel(model_name="qwen-vl-max-2025-04-08", enable_thinking=True, save_responses=True)

    # 5. Start the evaluation
    for i in range(len(dataset)):
        # Get the image and information of each sample
        sample = dataset.get_sample(i)
        image_path = sample["image_path"]

        # Invoke the model to make predictions.
        pred = model.predict(image_path, prompt)

        # Evaluate using the benchmark
        result = benchmark.evaluate(sample, pred)

        # Print the metrics.
        print(
            f"[{i+1}/{len(dataset)}] "
            f"Distance={result['metrics']['distance_km']:.1f} km | "
            f"Score={result['metrics']['score']} | "
            f"CountryCorrect={result['metrics']['country_correct']}"
        )

    # 6. Summarize the performance of the entire dataset on the benchmark
    summary = benchmark.summarize()

    # Print the summary
    print("\n=== Experiment Summary ===")
    print(f"Average Distance (km): {summary['avg_distance']:.1f}")
    print(f"Average Score: {summary['avg_score']:.1f}")
    print(f"Country Accuracy: {summary['country_acc']:.2%}")

    # 7. Save all the information from the experiment
    extra_meta={
        "model": model.model_name,
        "dataset": dataset.name, 
        "prompt_path": prompt_path
    }
    benchmark.save_results(extra_meta=extra_meta)


if __name__ == "__main__":
    main()
