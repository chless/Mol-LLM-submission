get-data:
	git clone https://huggingface.co/datasets/QizhiPei/BioT5_finetune_dataset biot5/data

get-rxn-data:
	rm -rf Mol-Instructions
	git lfs install
	git clone https://huggingface.co/datasets/zjunlp/Mol-Instructions
	unzip Mol-Instructions/data/Molecule-oriented_Instructions.zip -d biot5/data/tasks
	# remove the zip file
	rm -rf Mol-Instructions
	python biot5/reaction_prediction_preprocess.py


fine-tuning:
	task=$(task)
	data=$(task)
	model=biot5_base
	log_path=logs
	n_node=1
	n_gpu_per_node=4
	devices=1,2,3,4,5

	CUDA_VISIBLE_DEVICES=$(devices) torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} main.py \
		mode=ft \
		task=${task} \
		data=${task} \
		model=${model} \
		optim=${finetuning} \
		molecule_dict=dict/selfies_dict.txt \
		hydra.run.dir=${log_path} \
		seed=42 \
		pred.every_steps=100 logging.every_steps=100 \