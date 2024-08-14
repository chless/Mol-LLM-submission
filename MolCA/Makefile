ENV_NAME=molca-debug
init-env:
	conda create -n ${ENV_NAME} python=3.10 -y
	conda activate ${ENV_NAME}
	conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

	pip install git+https://github.com/thunlp/OpenDelta.git
	pip install rouge_score nltk ogb peft rdkit salesforce-lavis selfies deepchem Levenshtein evaluate
	pip install -U transformers==4.40.1 pytorch-lightning
	pip install deepspeed
	pip install torch_geometric
	pip install -U tensorboardX tensorboard

debug:
	conda create -n ${ENV_NAME} python=3.10 -y
	conda init bash
	conda activate ${ENV_NAME}