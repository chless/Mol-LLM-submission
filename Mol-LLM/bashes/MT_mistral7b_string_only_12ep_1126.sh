export TOKENIZERS_PARALLELISM=false;
file_name='MT_mistral7b_string_only_12ep_1126'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
data.raw_data_root=/data/Mol-LLM-v6 \
gnn.graph_encoder_ckpt=MoleculeSTM/molecule_model.pth \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
filename=$file_name \
trainer.logging_dir=/data/all_checkpoints

