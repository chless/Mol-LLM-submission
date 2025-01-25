
export TOKENIZERS_PARALLELISM=false;
file_name='MT_mistral7b_string+graph2string+graph_mdpo_4ep_lr5e-6_0125'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
data.data_tag=augmented \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.tune_gnn=true \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-post \
trainer.max_epochs=4 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=true \
ckpt_path="'/data/all_checkpoints/MT_mistral7b_string+graph2string+graph_mdpo_4ep_lr5e-6_0125/epoch=00-step=607.ckpt'" \
find_unused_parameters=true \
trainer.init_lr=0.000005 \
trainer.min_lr=0.0000005 \
trainer.warmup_lr=0.0000005
