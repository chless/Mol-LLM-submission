export TOKENIZERS_PARALLELISM=false;
file_name='PP_mistral7b_modality_random_mdpo_0108'
gpus="'1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=pp-super-small-v7.1 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.modality_randomization=true \
trainer.total_batch_size=512 \
trainer.max_epochs=10 \
trainer.val_check_interval=0.1

