[ -z "${task}" ] && task=molnet_hiv
[ -z "${model}" ] && model="biot5_base"
[ -z "${log_path}" ] && log_path="logs/molnet_hiv_0121"
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=6
[ -z "${devices}" ] && devices=0,1,2,3,4,5


export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=${devices}  torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} biot5/main.py \
    mode=ft \
    task=${task} \
    data=${task} \
    model=${model} \
    optim=finetuning_a100_80gb \
    molecule_dict=dict/selfies_dict.txt \
    hydra.run.dir=${log_path} \
    seed=42 \
    pred.every_steps=1000 logging.every_steps=100 \