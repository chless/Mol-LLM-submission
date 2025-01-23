
export TOKENIZERS_PARALLELISM=false;
file_name='qm9_3tasks_mistral7b_string+graph_mdpo_gnn_tuning_simpo0.3-b1.0-g1.0_0117'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=qm9_3tasks \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.total_batch_size=256 \
trainer.simpo_modality=string+graph \
trainer.max_epochs=10 \
trainer.val_check_interval=0.25 \
trainer.skip_sanity_check=true \
trainer.beta=1.0 \
trainer.gamma_beta_ratio=1.0 \
trainer.simpo_weight=0.3 \
ckpt_path=/data/all_checkpoints/qm9_3tasks_mistral7b_string+graph_mdpo_gnn_tuning_0117/last.ckpt


