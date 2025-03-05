export TOKENIZERS_PARALLELISM=false;
file_name='qm9_string_only_12ep_0228'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data.data_tag=qm9_0228 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.max_epochs=24 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string_only \
trainer.projector_type=mlp \
trainer.skip_sanity_check=false

