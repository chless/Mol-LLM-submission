export TOKENIZERS_PARALLELISM=false;
file_name='PP_mistral7b_modality_random_mdpo_simpo0.5_gnn_freeze_0111'
gpus="'4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=pp-super-small-v7.1 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM_freeze \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo \
trainer.total_batch_size=256 \
trainer.simpo_weight=0.5 \
trainer.modality_randomization=true \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.max_epochs=5 \
trainer.val_check_interval=0.25 \
trainer.skip_sanity_check=true \
ckpt_path="'/data/all_checkpoints/PP_mistral7b_modality_random_mdpo_simpo0.5_0111/last.ckpt'"

