export TOKENIZERS_PARALLELISM=false;
ckpt_path=/data/Mol-LLM-v6/logging_s
file_name='hug_test_s'
gpus="'0,1,2,3'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
data.raw_data_root=/data/Mol-LLM-v6 \
gnn.graph_encoder_ckpt=MoleculeSTM/molecule_model.pth \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
filename=$file_name \
trainer.logging_dir=${ckpt_path}/${filename}/all_checkpoints/

