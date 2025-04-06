
export TOKENIZERS_PARALLELISM=false;
file_name='qm9_string_only_test-v2'
gpus="'0,1,2,3'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=test \
data=multi_task \
data.data_tag=qm9_0211-v2 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-v2 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string_only \
trainer.total_batch_size=256 \
trainer.max_epochs=5 \
trainer.val_check_interval=0.25 \
trainer.skip_sanity_check=true \
trainer.batch_size=4 \
trainer.inference_batch_size=7 \
trainer.anc_chosen_weight=0.5 \
trainer.anc_reject_weight=0.5 \
trainer.sft_lambda=1.0 \
trainer.reject_lambda=2.0 \
trainer.min_lr=0.0000002 \
trainer.init_lr=0.000002 \
trainer.warmup_lr=0.0000002 \
trainer.train_molpo=false \
trainer.eval_molpo=false \
ckpt_path="'/data/all_checkpoints/string_only-resume/all_checkpoints/string_only-resume/epoch=11-val_total_loss=0.000.ckpt'"
