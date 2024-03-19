[ -z "${task}" ] && task=molnet_hiv
[ -z "${model}" ] && model="biot5_base"
[ -z "${log_path}" ] && log_path="logs/molnet_hiv_0121"


export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=$1  python biot5/main.py \
    mode=ft \
    task=${task} \
    data=${task} \
    model=${model} \
    optim.base_lr=$2 \
    hydra.run.dir=${log_path}_$2 \
    seed=42 \
    logging.every_steps=10 \
    pred.every_steps=10 eval.every_steps=10 \