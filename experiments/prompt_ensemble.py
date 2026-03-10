'''
This script runs a prompt ensemble inference experiment for the multimodal geolocation task.

In this experiment, we adopt a prompt ensemble inference strategy to improve 
the robustness and performance of geolocation predictions produced by MLLMs.

Instead of relying on a single prompt, the model is queried with several
prompt templates. Each prompt produces an independent prediction
containing a country estimate and geographic coordinates. The 
predictions are then aggregated using a country-aware aggregation
strategy to obtain the final prediction.

The experiment extends the basic evaluation pipeline.
In detail, the experiment precedure consists of the following steps:
    1. Load a geolocation dataset consisting of images and ground-truth geographic annotations.
    2. For each image, 
        (1) Feed it together with several prompts respectively into a multimodal large language 
            model (MLLM) to obtain structured geographic predictions (country, latitude, longitude).
        (2) Aggregate these predictions using a country-aware voting strategy combined with a medoid-based 
            coordinate selection to obtain a final prediction for this image.
    3. Evaluate the aggregated predictions using a benchmark module to compute quantitative metrics
        such as geographic distance, GeoGuessr-style scores, and country-level accuracy.

This approach is expected to improve prediction stability by combining complementary 
reasoning perspectives induced by different prompts.
'''

from tqdm import tqdm

from utils.countryaware_aggregation import countryaware_aggregation

from geodatasets.osv5m_dataset import Osv5mDataset
from geodatasets.fairlocator_dataset import FairLocatorDataset
from prompts.prompt_manager import PromptManager
from benchmarks.osv5m_adapter import OSV5MAdapter
from models.qwen import QwenModel
# from models.qwen_lab_instruct import QwenModel
# from models.qwen_lab_thinking import QwenModel


def main():
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
        for prompt in prompts:
            pred = model.predict(image_path, prompt)
            if pred is None:
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
    main()
