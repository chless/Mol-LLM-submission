export TOKENIZERS_PARALLELISM=false;
file_name='MT_test_mistral7b_string_only_12ep_0225'
gpus="'0,1,2,3'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=test \
data.data_tag=augmented \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-v2 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string_only \
trainer.eval_molpo=true \
ckpt_path="'/data/all_checkpoints/string_only-resume-last/epoch=11-val_total_loss=0.000.ckpt'" \
trainer.skip_sanity_check=false

