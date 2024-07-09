# Use the official Python base image
FROM python:3.10

# Create a new conda environment
RUN conda create -n molca python=3.10

# Activate the conda environment
RUN conda activate molca

# Install PyTorch and related packages
RUN conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

# Install PyG
RUN conda install pyg -c pyg

# Install additional Python packages
RUN pip install git+https://github.com/thunlp/OpenDelta.git
RUN pip install rouge_score nltk ogb peft rdkit salesforce-lavis selfies deepchem Levenshtein evaluate
RUN pip install -U transformers==4.40.1 pytorch-lightning
RUN pip install deepspeed

# Set the default command to run when the container starts
CMD ["bash"]
