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

from tqdm import tqdm

from geodatasets.single_test_sample import SingleSample
from geodatasets.osv5m_dataset import Osv5mDataset
from geodatasets.fairlocator_dataset import FairLocatorDataset
from prompts.prompt_manager import PromptManager
from benchmarks.geobench_adapter import GeoBenchAdapter
from benchmarks.geoscore import GeoScore
from benchmarks.osv5m_adapter import OSV5MAdapter
from models.dummy_model import DummyModel
from models.qwen import QwenModel
# from models.qwen_lab_instruct import QwenModel
# from models.qwen_lab_thinking import QwenModel

def main():
    # 1. Prepare the dataset
    # dataset = SingleSample()
    dataset = Osv5mDataset(max_samples=10)
    # dataset = FairLocatorDataset(dataset_name="Breadth", max_samples=None)

    # 2. Load the prompt
    prompt_manager = PromptManager()
    # prompt_name = "geobench"
    prompt_name = "explicit_CoT"
    # prompt_name = "light_reasoning"
    # prompt_name = "direct_prediction"
    # prompt_name = "expert_persona"
    prompt = prompt_manager.get_prompt(prompt_name)

    # 3. Initialize the benchmark
    # benchmark = GeoBenchAdapter()
    # benchmark = GeoScore()
    benchmark = OSV5MAdapter()

    # 4. Initialize the model
    # model = DummyModel()
    # model = QwenModel(model_name="qwen-vl-max-2025-04-08", enable_thinking=False, save_responses=True)
    model = QwenModel(model_name="qwen3-vl-plus", enable_thinking=True, save_responses=True)
    # model = QwenModel(model_name="Qwen3-VL-8B-Instruct", enable_thinking=False, save_responses=True)
    # model = QwenModel(model_name="Qwen3-VL-8B-Thinking", enable_thinking=False, save_responses=True)

    # 5. Start the evaluation
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        # Get the image and information of each sample
        sample = dataset.get_sample(i)
        image_path = sample["image_path"]

        # Invoke the model to make predictions.
        pred = model.predict(image_path, prompt)

        # Evaluate using the benchmark
        result = benchmark.evaluate(sample, pred)

    # 6. Summarize the performance of the entire dataset on the benchmark
    summary = benchmark.summarize()

    # 7. Save all the information from the experiment
    extra_meta={
        "model": model.model_name,
        "dataset": dataset.name, 
        "prompt_name": prompt_name
    }
    benchmark.save_results(extra_meta=extra_meta)


if __name__ == "__main__":
    main()
