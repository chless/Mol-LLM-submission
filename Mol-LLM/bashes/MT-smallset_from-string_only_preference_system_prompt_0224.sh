
export TOKENIZERS_PARALLELISM=false;
file_name='MT-smallset_from-string_only_preference_system_prompt_0224'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=ft \
data=multi_task \
data.data_tag=ablation_0224_augmented \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-v2 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
trainer.sl_noise_ratio=0.3 \
trainer.apply_preference_system_prompt=true \
pretrained_ckpt_path="'/data/all_checkpoints/string_only-resume-last/epoch=11-val_total_loss=0.000.ckpt'" \
