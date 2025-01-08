export TOKENIZERS_PARALLELISM=false;
file_name='PP_mistral7b_string+graph_0108'
gpus="'5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=pp-v7.1 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.modality_randomization=false \
trainer.max_epochs=4

