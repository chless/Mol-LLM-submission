export TOKENIZERS_PARALLELISM=false;
file_name=MT_from-string_only_qformer_pretraining_1ep_0430
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
trainer=mistral7b_80gb_llava_pretraining \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string_only_12ep_0415/epoch=11-step=40991.ckpt'"
