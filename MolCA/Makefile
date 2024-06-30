ENV_NAME=molca-debug
init-env:
	conda create -n ${ENV_NAME} python=3.10 -y
	conda activate ${ENV_NAME}
	conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
	conda install pyg -c pyg -y
	pip install git+https://github.com/thunlp/OpenDelta.git
	pip install rouge_score nltk ogb peft rdkit salesforce-lavis selfies deepchem
	pip install -U transformers==4.40.1 pytorch-lightning
	pip install deepspeed
	pip install torch_geometric
	pip install -U tensorboardX tensorboard

debug:
	conda create -n ${ENV_NAME} python=3.10 -y
	conda init bash
	conda activate ${ENV_NAME}