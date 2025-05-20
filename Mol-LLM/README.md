# Introduction

This repository provide necessary code, model, and testset to reproduce results in the paper "Mol-LLM: Multimodal Generalist Molecular LLM with Improved Graph Utilization", to provide rich information during rebuttal process.

# Access to Model and Dataset
The model checkpoitns and testsets are available via [GDrive](https://drive.google.com/drive/folders/1lgPgkcdA_EMX5iQlhyJa-DgBEO4tBxa1?usp=sharing).
* Mol-LLM [[GDrive]](https://drive.google.com/file/d/1Oun_iGZah61T9bP3mGqJPCANCtGBKxgs/view?usp=sharing)
* Mol-LLM (w/o Graph) [[GDrive]](https://drive.google.com/file/d/12abcNyngE1ByrDAduB1zGE-lwuCNFRnl/view?usp=sharing)
* Testset [[GDrive]](https://drive.google.com/drive/folders/1D6nqfwmc5IxG9DT6NrPtFLCvtgNasJbG?usp=sharing)


# Installation
For easy and fast reproduction, all environments are built based on `docker` and `Makefile`.
1. Build `docker` image using `Makefile`: `make build-image`
2. Before initialize `docker` container, set following volume mounting path in `Makefile`
   * `REPO_PATH=/home/{user_name}/text-mol` : The path of the repository
   * `CACHE_PATH=/home/{user_name}/.cache` : Huggingface cache path
   * `IMAGE_NAME_TAG={user_name}/mol-llm:v1` : The name of the built docker image
3. FInally, initialize docker container using `Makefile`: `make init-container`

# Reproduction of results
* To reproduce performance of `Mol-LLM` through Main Table 1-4, run the following command:  `bash /text-mol/Mol-LLM/bashes/mol-llm_test.sh "'{your_gpu_devices}'"`
  * For example, if you want to run evaluation with `GPU=0,1`, then input `your_gpu_devices=0,1`
* To reproduce performance of `Mol-LLM (w/o Graph)` through Main Table 1-4, run the following command: `bash /text-mol/Mol-LLM/bashes/mol-llm_wo_graph_test.sh "'{your_gpu_devices}'"`



