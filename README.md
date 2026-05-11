# Introduction

This repository provide necessary code, model, and test dataset to reproduce results in the paper "Mol-LLM: Multimodal Generalist Molecular LLM with Improved Graph Utilization", to support rebuttal process by providing detailed information.

# Access to Model and Dataset
For reproducibility, the **model checkpoints and test dataset** are available via [GDrive](https://drive.google.com/drive/folders/1bEl9dB2SD9aQ3hhk2Yg8o5oTX0Hqjj9x?usp=sharing).
The corresponding model card and dataset cards are available on [Huggingface](https://huggingface.co/collections/KU-AGI/mol-llm), while download is only available for test dataset.
After acceptance, the **model checkpoints and train and test dataset** will be released via [Huggingface](https://huggingface.co/collections/KU-AGI/mol-llm).

* Mol-LLM [[GDrive]](https://drive.google.com/file/d/1CxLY4rOGiHMwvUvyhUO-ovT2ehhtcDGM/view?usp=sharing)[[Huggingface]](https://huggingface.co/KU-AGI/Mol-LLM)
* Mol-LLM (w/o Graph) [[GDrive]](https://drive.google.com/file/d/1jjPsSElKNfakZ9J_Hk82pL9gxipzpLVA/view?usp=sharing)[[Huggingface]](https://huggingface.co/KU-AGI/Mol-LLM-wo-graph)
* Testset [[GDrive]](https://drive.google.com/drive/folders/1bvdwcSffydgX5ErUaIK9EskTd2TSjupR?usp=sharing)[[Huggingface]](https://huggingface.co/datasets/KU-AGI/Mol-LLM-testset).
After downloading the checkpoints and test dataset, adjust the path to each file by following the instructions in the Installation.


# Installation
For easy and fast reproduction, all environments are built based on `docker` and `Makefile`.
Installation would take around half an hour in total.
1. Build `docker` image using `Makefile`: `make build-image`
2. Before initialize `docker` container, set following volume mounting path in `Makefile`
   * `REPO_PATH=/home/{user_name}/text-mol` : The path of the repository
   * `CACHE_PATH=/home/{user_name}/.cache` : Huggingface cache path
   * `IMAGE_NAME_TAG={user_name}/mol-llm:v1` : The name of the built docker image
3. FInally, initialize docker container using `Makefile`: `make init-container`

# Reproduction of results
* To reproduce performance of `Mol-LLM` reported in Main Tables 1–4, run the following command (around 2 hours on an 8×A100 server):  `bash /text-mol/Mol-LLM/bashes/mol-llm_test.sh "'{your_gpu_devices}'"`
  * For example, if you want to run evaluation with `GPU=0,1`, then input `your_gpu_devices=0,1`
* To reproduce the results of `Mol-LLM (w/o Graph)` through Main Table 1-4, run the following command (around 2 hours on an 8×A100 server): `bash /text-mol/Mol-LLM/bashes/mol-llm_wo_graph_test.sh "'{your_gpu_devices}'"`
* Demo inference example (about 10–30 seconds for a single example, on a A100 server):
  ```text
  # Demo input: log-solubility prediction from a SELFIES+GRAPH prompt
  <s>[INST] You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.

  What is the log solubility of <SELFIES>...</SELFIES><GRAPH>...</GRAPH> in water? [/INST]
  
  # Expected output:
  <FLOAT> <|-|><|4|><|.|><|4|><|7|><|2|><|0|> </FLOAT>
  ```



