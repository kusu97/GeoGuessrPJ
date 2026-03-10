'''
This script runs a self-consistency experiment for the geolocation task.

In this experiment, we adopt a self-consistency decoding strategy to improve 
the robustness and performance of geolocation predictions produced by MLLMs.

The experiment extends the basic evaluation pipeline, introducing a country-aware 
aggregation to implement the self-consistency strategy.
In detail, the experiment precedure consists of the following steps:
    1. Load a geolocation dataset consisting of images and ground-truth geographic annotations.
    2. Feed each image together with a specified prompt into a multimodal large language model (MLLM)
        for K times to obtain K structured geographic predictions (country, latitude, longitude).
    3. For each image, aggregate the K predictions using a country-aware voting strategy combined 
        with a medoid-based coordinate selection to obtain the final prediction.
    4. Evaluate the aggregated predictions using a benchmark module to compute quantitative metrics
        such as geographic distance, GeoGuessr-style scores, and country-level accuracy.

By leveraging multiple stochastic predictions and aggregating them through country-aware 
voting and medoid selection, self-consistency is expected to provide a simple yet effective way 
to improve the stability and accuracy of geolocation predictions.
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


K = 3       # Consistency@K

def main():
    # 1. Prepare the dataset
    dataset = Osv5mDataset(max_samples=200)
    # dataset = FairLocatorDataset(dataset_name="Breadth", max_samples=None)

    # 2. Load the prompt
    prompt_manager = PromptManager()
    prompt_name = "explicit_CoT"
    # prompt_name = "light_reasoning"
    # prompt_name = "direct_prediction"
    # prompt_name = "expert_persona"
    prompt = prompt_manager.get_prompt(prompt_name)

    # 3. Initialize the benchmark
    benchmark = OSV5MAdapter()

    # 4. Initialize the model (Note: model temperature MUST > 0)
    model = QwenModel(model_name="qwen3-vl-plus", temperature=0.8)
    # model = QwenModel(model_name="Qwen3-VL-8B-Instruct", temperature=0.8)
    # model = QwenModel(model_name="Qwen3-VL-8B-Thinking", temperature=0.8)

    # 5. Start the evaluation
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        # Get the image and information of each sample
        sample = dataset.get_sample(i)
        image_path = sample["image_path"]

        # Invoke the model to make predictions for K times
        predictions = []
        for i in range(K):
            pred = model.predict(image_path, prompt)
            if pred is None:
                continue
            predictions.append(pred)

        # Perform country-aware aggregation to get the final prediction
        if predictions == []:
            # All K prediction attempts fail
            final_pred = None
        else:
            final_pred = countryaware_aggregation(predictions)

        # Evaluate using the benchmark
        result = benchmark.evaluate(sample, final_pred)

    # 6. Summarize the performance of the entire dataset on the benchmark
    summary = benchmark.summarize()

    # 7. Save all the information from the experiment
    extra_meta={
        "experiment": "self-consistency",
        "K": K,
        "model_info": model.info,
        "dataset": dataset.name, 
        "prompt_name": prompt_name
    }
    benchmark.save_results(extra_meta=extra_meta)


if __name__ == "__main__":
    main()
