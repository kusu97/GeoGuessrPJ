'''
This script runs a multi-agent geolocation pipeline implemented using AutoGen.

The experiment extends the basic evaluation pipeline.
In detail, the experiment precedure consists of the following steps:
    1. Load a geolocation dataset consisting of images and ground-truth geographic annotations.
    2. Feed each image into a multi-agent pipeline to obtain structured geographic predictions 
    (country, latitude, longitude).
    3. Evaluate the model predictions using a benchmark module to compute quantitative metrics
    such as geographic distance, GeoGuessr-style scores, and country-level accuracy.

This implementation serves as the base framework for all multi-agent experiments, 
including multi-agent collaboration, debate-based reasoning and self-refinement strategies.
'''

from tqdm import tqdm
import asyncio

from geodatasets.osv5m_dataset import Osv5mDataset
from geodatasets.fairlocator_dataset import FairLocatorDataset
from benchmarks.osv5m_adapter import OSV5MAdapter
from models.multi_agent_collaboration import MultiAgentCollaboration
from models.multi_agent_debate import MultiAgentDebate
from models.self_refine import SelfRefine

def main():
    # 1. Prepare the dataset
    dataset = Osv5mDataset(max_samples=200)
    # dataset = FairLocatorDataset(dataset_name="Breadth", max_samples=None)

    # 2. Initialize the benchmark
    benchmark = OSV5MAdapter()

    # 3. Initialize the multi-agent pipeline
    client_info = {
        "model": "YOUR MODEL NAME",
        "api_key": "YOUR API KEY",
        "base_url": "YOUR BASE URL",
        "model_info": {
            "vision": "BOOL",
            "function_calling": "BOOL",
            "json_output": "BOOL",
            "structured_output": "BOOL",
            "family": "YOUR MODEL FAMILY"
        },
        "temperature": 0.0,
        "max_tokens": 2000
    }
    # Note: The client_info for each agent can be different
    pipeline = MultiAgentCollaboration(client_info, client_info, client_info)
    # pipeline = MultiAgentDebate(client_info, client_info, client_info)
    # pipeline = SelfRefine(client_info)

    # 4. Start the evaluation
    async def run_evaluation(dataset, benchmark):

        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            # Get the image and information of each sample
            sample = dataset.get_sample(i)
            image_path = sample["image_path"]

            # Reset the memory of the agents
            await pipeline.reset()

            # Invoke the pipeline to make predictions
            pred = await pipeline.predict(image_path, "")

            # Evaluate using the benchmark
            result = benchmark.evaluate(sample, pred)
    
    asyncio.run(run_evaluation(dataset, benchmark))

    # 5. Summarize the performance of the entire dataset on the benchmark
    summary = benchmark.summarize()

    # 6. Save all the information from the experiment
    extra_meta={
        "experiment": "multi-agent",
        "pipeline": pipeline.pipeline_name,
        "client_info": pipeline.all_client_info,
        "dataset": dataset.name
    }
    benchmark.save_results(extra_meta=extra_meta)


if __name__ == "__main__":
    main()
