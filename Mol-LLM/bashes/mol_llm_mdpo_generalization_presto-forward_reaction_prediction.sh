
export TOKENIZERS_PARALLELISM=false;
file_name='mol_llm_mdpo_generalization_presto-forward_reaction_prediction'
gpus="'0'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=test \
data=multi_task \
data.data_tag=presto-forward_reaction_prediction \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.tune_gnn=true \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-post \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
ckpt_path="'/data/all_checkpoints/MT_mistral7b_string+graph2string+graph_mdpo_5ep_lr_0124/epoch=02-step=6685.ckpt'" \
trainer.eval_simpo=false \
trainer.inference_batch_size=5
