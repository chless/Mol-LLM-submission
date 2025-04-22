export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn=gine_tokengt
file_name=MT_mistral7b_string+graph_${gnn}_12ep_0423

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
gnn=${gnn} \
trainer=mistral7b_80gb \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string+graph_gine_qformer_pretraining_1ep_0415/epoch=00-step=3267.ckpt'" \

