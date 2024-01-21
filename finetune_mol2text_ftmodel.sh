[ -z "${task}" ] && task=mol2text
[ -z "${model}" ] && model="biot5_base_mol2text"
[ -z "${log_path}" ] && log_path="logs/mol2text_0121"
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=1
[ -z "${devices}" ] && devices=4


export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=${devices}  python3 biot5/main.py \
    mode=ft \
    task=${task} \
    data=${task} \
    model=${model} \
    hydra.run.dir=${log_path} \
    pred.every_steps=1000 logging.every_steps=100 \