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



