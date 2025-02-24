export TOKENIZERS_PARALLELISM=false;
file_name='MT_test_mistral7b_string+graph_12ep_0224'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data.data_tag=augmented \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
ckpt_path="'/data/all_checkpoints/hug_test_sg_cls/all_checkpoints/hug_test_sg_cls/every_epochs/epoch=11-val_loss=0.000000.ckpt'" \
trainer.skip_sanity_check=false

