export TOKENIZERS_PARALLELISM=false;
file_name='MT_mistral7b_string_only_12ep_1126'
gpus="'1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
ckpt_path="'/data/all_checkpoints/MT_mistral7b_string_only_12ep_1126/last.ckpt'" \
mode=test

