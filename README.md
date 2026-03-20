# GeoGuessrPJ: GeoGuessr with Multimodal Large Language Models

This repository is an experimental project for the research topic  
**“First-Person Streaming Visual Agents — Starting from GeoGuessr”**.

The goal of this project is to build a **complete and reproducible pipeline** that evaluates the geolocation ability of multimodal large language models (MLLMs) on GeoGuessr-style tasks. In addition, we aim to further explore potential improvement strategies.

---

## 1. Project Overview

GeoGuessr is a visual geolocation task where a multi-modal large language model must infer the geographic location of a place based on a single first-person street-view image.

In this project, we:
- Use **multi-modal large language models via online APIs**
- Perform **location prediction from street-view images**
- Evaluate predictions using **standard geolocation benchmarks**
- Explore and validate **potential optimization schemes**

---

## 2. Task Definition

### Input
- A single street-view image (first-person perspective)

### Output
- A predicted geographic location:
  - Latitude
  - Longitude
  - Country (optional)

### Evaluation
- Distance-based geolocation metrics (e.g. average error in kilometers, radius accuracy)
- Benchmark-provided scores (e.g. GeoScore)
- Administrative accuracy (e.g. country accuracy)   

---

## 3. Project Structure

```
.
├── geodatasets/ # GeoGuessr-style street-view datasets
├── models/ # Model interface codes and multi-agent pipelines
├── benchmarks/ # Geolocation benchmarks
├── prompts/ # Different prompts
├── experiments/ # Basic and exploratory experiments
│ └── run_basic_experiment.py
│ └── self_consistency.py
│ └── multi-agent.py
│ └── ...
├── records/ # Records of experiment results
├── utils/ # Utility scripts
├── README.md
└── environment.yml
```

---

## 4. Models

This project primarily uses **online multimodal model APIs**, such as:
- Qwen-vl series

The choice of API allows rapid experimentation without local deployment constraints.

Besides, the **multi-agent pipelines** (implemented using AutoGen) are also defined here, including:
- multi-agent collaboration
- multi-agent debate
- self-refine

---

## 5. Benchmarks

Standard **GeoGuessr-style geolocation benchmarks** are used, including:
- GeoBench
- GeoScore
- Osv-5m

The benchmark is executed locally, while model inference is performed via API calls.

---

## 6. Datasets

This project uses **GeoGuessr-style street-view image datasets** for visual geolocation experiments, such as:
- FairLocator
- Osv-5m

---

## 7. Experiments

The **basic experimental pipeline** (run_basic_experiment.py) follows these steps:

1. Load a street-view dataset
2. Send image + prompt to a multimodal model API
3. Parse the model’s textual output to obtain geographic information
4. Run benchmark evaluation
5. Analyze and save results

Some **preliminary comparative experiments** (e.g., studies on prompt design and model choices) were tested using this pipeline structure, and **all extended experiments** (e.g., self-consistency and multi-agent collaboration) are built upon it.

**Further optimization schemes** include:

- multi-agent experiments(including multi-agent collaboration, multi-agent debate and iterative self-refine)
- multi-view input
- prompt ensemble
- self-consistency
- PESC (combined Prompt Ensemble and Self-Consistency)

For implementation details of these strategies, please refer to the corresponding source code.