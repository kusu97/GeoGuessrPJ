'''
This script runs a prompt ensemble inference experiment 
using existing results from previous basic experiments.

Note:
1. The results produced by this script are fully compatible 
    in format with those from "prompt_ensemble.py".
2. The existing results used must match the settings for
    the model, dataset, prompts, and benchmark.
3. The existing results used must be from basic experiments.
'''

import json
from tqdm import tqdm

from utils.countryaware_aggregation import countryaware_aggregation
from models.base import Prediction

from geodatasets.osv5m_dataset import Osv5mDataset
from geodatasets.fairlocator_dataset import FairLocatorDataset
from prompts.prompt_manager import PromptManager
from benchmarks.osv5m_adapter import OSV5MAdapter
from models.qwen import QwenModel
# from models.qwen_lab_instruct import QwenModel
# from models.qwen_lab_thinking import QwenModel


def main(existing_results_path):
    # 0. Get the existing results
    existing_results = []
    for path in existing_results_path:
        with open(path) as f:
            existing_result = json.load(f)
        existing_results.append(existing_result)
    
    # 1. Prepare the dataset
    dataset = Osv5mDataset(max_samples=200)
    # dataset = FairLocatorDataset(dataset_name="Breadth", max_samples=None)

    # 2. Load the prompts
    prompt_manager = PromptManager()

    prompt_names = ["explicit_CoT", "light_reasoning", "direct_prediction"]
    # prompt_names = ["explicit_CoT", "light_reasoning", "direct_prediction", "expert_persona"]

    prompts = [prompt_manager.get_prompt(prompt_name) for prompt_name in prompt_names]

    # 3. Initialize the benchmark
    benchmark = OSV5MAdapter()

    # 4. Initialize the model (Note: temperature = 0 suggested)
    model = QwenModel(model_name="qwen3-vl-plus")
    # model = QwenModel(model_name="Qwen3-VL-8B-Instruct")
    # model = QwenModel(model_name="Qwen3-VL-8B-Thinking")

    # 5. Start the evaluation
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        # Get the image and information of each sample
        sample = dataset.get_sample(i)
        image_path = sample["image_path"]

        # Invoke the model to make predictions with multiple prompts
        predictions = []
        for existing_result in existing_results:
            for result in existing_result["results"]:
                if result["image_path"] == image_path:
                    pred_dict = result["pred"]
                    pred = Prediction(
                        country=pred_dict["country"],
                        lat=pred_dict["lat"],
                        lng=pred_dict["lng"]
                    )
                    break
            else:
                continue
            predictions.append(pred)

        # Perform country-aware aggregation to get the final prediction
        if predictions == []:
            # All prediction attempts fail
            final_pred = None
        else:
            final_pred = countryaware_aggregation(predictions)

        # Evaluate using the benchmark
        result = benchmark.evaluate(sample, final_pred)

    # 6. Summarize the performance of the entire dataset on the benchmark
    summary = benchmark.summarize()

    # 7. Save all the information from the experiment
    extra_meta={
        "experiment": "prompt ensemble",
        "prompt_names": prompt_names,
        "model_info": model.info,
        "dataset": dataset.name
    }
    benchmark.save_results(extra_meta=extra_meta)


if __name__ == "__main__":
    existing_results = ["your_result_path1", "your_result_path2", "..."]
    main(existing_results)
