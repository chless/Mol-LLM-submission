export TOKENIZERS_PARALLELISM=false;
file_name='PP_mistral7b_string+graph_gnn_tuning_0112'
gpus="'2,3'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=pp-super-small-v7.1 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.total_batch_size=256 \
trainer.modality_randomization=false \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.max_epochs=5 \
trainer.val_check_interval=0.25 \
trainer.skip_sanity_check=true

