

fine-tuning:
	task=$(task)
	data=$(task)
	n_node=1
	n_gpu_per_node=4
	log_path=logs
	model=biot5_base

	torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} main.py \
		mode=ft \
		task=${task} \
		data=${task} \
		model=${model} \
		optim=${finetuning} \
		molecule_dict=dict/selfies_dict.txt \
		hydra.run.dir=${log_path} \
		seed=42 \
		pred.every_steps=100 logging.every_steps=100 \