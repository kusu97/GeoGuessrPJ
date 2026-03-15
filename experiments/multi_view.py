'''
This script runs a multi-view reasoning experiment for the geolocation task.

In this experiment, instead of predicting the location from a single street-view image, 
the model analyzes several street views captured from nearby positions within one city.

The experiment extends the basic evaluation pipeline.
In detail, the experiment precedure consists of the following steps:
    1. Load a geolocation dataset consisting of images and ground-truth geographic annotations.
    2. Feed several images from the same city together with a specified prompt into a multimodal large 
        language model (MLLM) to obtain structured geographic predictions (country, latitude, longitude).
    3. Evaluate the predictions using a benchmark module to compute quantitative metrics such as 
        geographic distance, GeoGuessr-style scores, and country-level accuracy.

By leveraging more geographic information extracted from multiple images of the same city, 
multi-view reasoning is expected to improve the geolocation accuracy. 
'''

from tqdm import tqdm

from geodatasets.multi_view_fairlocator import MultiViewFairLocatorDataset
from prompts.prompt_manager import PromptManager
from benchmarks.geobench_adapter import GeoBenchAdapter
from benchmarks.geoscore import GeoScore
from benchmarks.osv5m_adapter import OSV5MAdapter
from models.multi_view_qwen import QwenModel
# from models.multi_view_qwen_lab_instruct import QwenModel
# from models.multi_view_qwen_lab_thinking import QwenModel

def main():
    # 1. Prepare the dataset
    dataset = MultiViewFairLocatorDataset(dataset_name="Breadth", num_views=10, max_samples=2)

    # 2. Load the prompt
    prompt_manager = PromptManager()
    # prompt_name = "explicit_CoT"
    prompt_name = "light_reasoning"
    # prompt_name = "direct_prediction"
    # prompt_name = "expert_persona"
    prompt = prompt_manager.get_prompt(prompt_name)

    # Replace the description of the prompt
    old_desc = "You are participating in a geolocation challenge. Based on the provided image:"
    new_desc = f"You are participating in a geolocation challenge.\n\nYou are given {dataset.num_views} street view images captured from the same city.\n\nBased on the provided images:"
    prompt = prompt.replace(old_desc, new_desc)

    # 3. Initialize the benchmark
    # benchmark = GeoBenchAdapter()
    # benchmark = GeoScore()
    benchmark = OSV5MAdapter()

    # 4. Initialize the model
    model = QwenModel(model_name="qwen-vl-max-2025-04-08")
    # model = QwenModel(model_name="qwen3-vl-plus", enable_thinking=True)
    # model = QwenModel(model_name="Qwen3-VL-8B-Instruct")
    # model = QwenModel(model_name="Qwen3-VL-8B-Thinking")

    # 5. Start the evaluation
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        # Get the images and information of each sample
        sample = dataset.get_sample(i)
        image_paths = sample["image_path"]

        # Invoke the model to make predictions.
        pred = model.predict(image_paths, prompt)

        # Evaluate using the benchmark
        result = benchmark.evaluate(sample, pred)

    # 6. Summarize the performance of the entire dataset on the benchmark
    summary = benchmark.summarize()

    # 7. Save all the information from the experiment
    extra_meta={
        "experiment": "multi-view",
        "num_views": dataset.num_views,
        "model_info": model.info,
        "dataset": dataset.name, 
        "prompt_name": prompt_name
    }
    benchmark.save_results(extra_meta=extra_meta)


if __name__ == "__main__":
    main()
