[ -z "${task}" ] && task=forward_reaction_prediction
[ -z "${model}" ] && model="biot5_base"
[ -z "${log_path}" ] && log_path="logs/forward_reaction_prediction_0130_5e-5"
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=4
[ -z "${devices}" ] && devices="0,1,2,3"

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=${devices}  torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} --master_port=25511 biot5/main.py \
    mode=ft \
    task=${task} \
    data=${task} \
    model=${model} \
    optim=finetuning_lr_5e-5_forward_reaction_prediction \
    hydra.run.dir=${log_path} \
    logging.every_steps=10 \
    pred.every_steps=500 eval.every_steps=500 \
