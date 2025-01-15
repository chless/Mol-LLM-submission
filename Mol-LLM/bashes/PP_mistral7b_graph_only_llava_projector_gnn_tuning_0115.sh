export TOKENIZERS_PARALLELISM=false;
file_name='PP_mistral7b_graph_only_llava_projector_gnn_tuning_0115'
gpus="'1,2,3'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=pp-super-small-v7.1 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.projector_type=mlp \
trainer.llava_pretraining=1 \
trainer.second_stage_start_epoch=1 \
trainer.total_batch_size=256 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=graph_only \
trainer.max_epochs=6 \
trainer.val_check_interval=0.25 \
trainer.skip_sanity_check=true

