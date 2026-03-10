'''
This script runs a combined Prompt Ensemble and Self-Consistency inference 
experiment for the multimodal geolocation task.

For each input image, the model is queried using multiple prompt
templates to encourage diverse reasoning strategies (prompt ensemble).
For every prompt, multiple predictions are sampled using stochastic
decoding (self-consistency). All predictions are then aggregated using
a country-aware strategy: majority voting over predicted countries
followed by medoid selection among the remaining coordinates.

This approach introduces diversity at both the prompt level and the
sampling level, thus expected to improve the robustness and stability 
of geolocation predictions.
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

    # 2. Load the prompts
    prompt_manager = PromptManager()

    prompt_names = ["explicit_CoT", "light_reasoning", "direct_prediction"]
    # prompt_names = ["explicit_CoT", "light_reasoning", "direct_prediction", "expert_persona"]

    prompts = [prompt_manager.get_prompt(prompt_name) for prompt_name in prompt_names]

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

        # Invoke the model to make predictions with multiple prompts, each for K times
        predictions = []
        for prompt in prompts:
            for i in range(K):
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
        "experiment": "PESC (combined Prompt Ensemble and Self-Consistency)",
        "K": K,
        "prompt_names": prompt_names,
        "model_info": model.info,
        "dataset": dataset.name
    }
    benchmark.save_results(extra_meta=extra_meta)


if __name__ == "__main__":
    main()
