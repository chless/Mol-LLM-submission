get-data:
	git clone https://huggingface.co/datasets/QizhiPei/BioT5_finetune_dataset biot5/data


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