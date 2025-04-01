
export TOKENIZERS_PARALLELISM=false;
file_name='MT_molpo_0402'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=ft \
data=molpo \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_molpo \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
trainer.sl_noise_ratio=0.0 \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string_only_12ep_0224/epoch=11-step=61512.ckpt'"
