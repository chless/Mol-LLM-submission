export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=MT_string+graph_12ep_0430
max_epochs=12

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=molpo \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string+graph_gine_qformer_pretraining_1ep_0415/epoch=00-step=3267.ckpt'"
