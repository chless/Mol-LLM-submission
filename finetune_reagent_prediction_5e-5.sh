[ -z "${task}" ] && task=reagent_prediction
[ -z "${model}" ] && model="biot5_base"
[ -z "${log_path}" ] && log_path="logs/reagent_prediction_0205_5e-5"
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=3
[ -z "${devices}" ] && devices="3,4,5"

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=${devices}  torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} --master_port=25512 biot5/main.py \
    mode=ft \
    task=${task} \
    data=${task} \
    model=${model} \
    optim=finetuning_lr_5e-5_reagent_prediction \
    hydra.run.dir=${log_path} \
    logging.every_steps=1 \
    pred.every_steps=1 eval.every_steps=1 \
