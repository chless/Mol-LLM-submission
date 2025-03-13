
export TOKENIZERS_PARALLELISM=false;
file_name='qm9_molpo_from-string+graph'
gpus="'0,1,2,3'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=ft \
data=multi_task \
data.data_tag=qm9_homo_0211_augmented \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-v2 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.total_batch_size=256 \
trainer.max_epochs=5 \
trainer.val_check_interval=0.2 \
trainer.skip_sanity_check=true \
trainer.batch_size=4 \
trainer.inference_batch_size=7 \
trainer.reject_lambda=3.0 \
pretrained_ckpt_path="'/data/all_checkpoints/hug_test_sg_cls/all_checkpoints/hug_test_sg_cls/every_epochs/epoch=11-val_loss=0.000000.ckpt'"
