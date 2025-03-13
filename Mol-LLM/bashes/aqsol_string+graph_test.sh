
export TOKENIZERS_PARALLELISM=false;
file_name='aqsol_molpo-string+graph_test'
gpus="'6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=test \
data=multi_task \
data.data_tag=aqsol_0212 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-v2 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.train_simpo=false \
trainer.eval_simpo=false \
ckpt_path="'/data/all_checkpoints/MT_from-string+graph_0211/epoch=00-step=486.ckpt'"
