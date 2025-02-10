
export TOKENIZERS_PARALLELISM=false;
file_name='qm9_molpo_anc-sft0.5-rej0.5'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=post-ft \
data=multi_task \
data.data_tag=qm9_0207_augmented \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-v2 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.total_batch_size=256 \
trainer.max_epochs=5 \
trainer.val_check_interval=0.25 \
trainer.skip_sanity_check=true \
trainer.batch_size=4 \
trainer.inference_batch_size=7 \
trainer.anc_sft_weight=0.5 \
trainer.anc_reject_weight=0.0 \
trainer.sft_lambda=1.0 \
trainer.reject_lambda=2.0 \
trainer.anchor_clamp=-1 \
pretrained_ckpt_path="'/data/all_checkpoints/hug_test_sg_cls/all_checkpoints/hug_test_sg_cls/every_epochs/epoch=11-val_loss=0.000000.ckpt'"
