
export TOKENIZERS_PARALLELISM=false;
file_name='qm9_4tasks_mistral7b_string_only_0117'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=qm9_4tasks \
data.data_tag=qm9_4tasks_offline \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string_only \
trainer.total_batch_size=256 \
trainer.max_epochs=10 \
trainer.val_check_interval=0.25 \
trainer.skip_sanity_check=true \
trainer.batch_size=6 \
trainer.inference_batch_size=12
