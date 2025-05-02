export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=MT_from-string_only_12ep_string+graph_6ep_0502
max_epochs=6

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_from-string_only_qformer-gine_tokengt_pretraining_1ep_0501/epoch=00-step=3415_lora_compensated.ckpt'" \
trainer.init_lr=0.00004 \
trainer.warmup_lr=0.000004 \
trainer.val_check_interval=0.20 \
trainer.batch_size=8

