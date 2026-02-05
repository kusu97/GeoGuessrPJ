# GeoGuessrPJ: GeoGuessr with Multimodal Large Language Models

This repository is an experimental project for the research topic  
**“First-Person Streaming Visual Agents — Starting from GeoGuessr”**.

The goal of this project is to build a **complete and reproducible pipeline** that evaluates the geolocation ability of multimodal large language models (MLLMs) on GeoGuessr-style tasks. In addition, we aim to further explore potential improvement strategies.

---

## 1. Project Overview

GeoGuessr is a visual geolocation task where a model must infer the geographic location of a place based on a single first-person street-view image.

In this project, we:
- Use **multimodal large language models via online APIs**
- Perform **location prediction from street-view images**
- Evaluate predictions using **standard geolocation benchmarks**

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
├── models/ # Model interface codes
├── benchmarks/ # Geolocation benchmark codes
├── prompts/ # Different prompts
├── experiments/
│ └── run_basic_experiment.py # Run basic experiment
├── records/ # Records of experiment results
├── references/ # Other project scripts used as references
├── scripts/ # Various scripts to be refined and archived
├── utils/ # Utility scripts
├── README.md
└── environment.yml
```

> Note: The structure may be modified during future experiments.

---

## 4. Models

This project primarily uses **online multimodal model APIs**, such as:
- qwen3-vl-plus

The choice of API allows rapid experimentation without local deployment constraints.

---

## 5. Benchmarks

Standard **GeoGuessr-style geolocation benchmarks** are used, including:
- GeoBench
- GeoScore
- Osv-5m

The benchmark is executed **locally**, while model inference is performed via API calls.

---

## 6. Datasets

This project uses **GeoGuessr-style street-view image datasets** for visual geolocation experiments, such as:
- FairLocator
- Osv-5m

---

## 7. Basic Workflow

The basic experimental pipeline follows these steps:

1. Load a street-view dataset
2. Send image + prompt to a multimodal model API
3. Parse the model’s textual output to obtain geographic information
4. Run benchmark evaluation
5. Analyze and save results
