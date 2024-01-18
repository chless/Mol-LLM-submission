[ -z "${task}" ] && task=text2mol
[ -z "${model}" ] && model="biot5_base"
[ -z "${log_path}" ] && log_path="logs/text2mol_0114"
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=4
[ -z "${devices}" ] && devices=0,1,2,3

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=${devices}  torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} biot5/main.py \
    mode=ft \
    task=${task} \
    data=${task} \
    model=${model} \
    optim=finetuning_a100_80gb \
    hydra.run.dir=${log_path} \