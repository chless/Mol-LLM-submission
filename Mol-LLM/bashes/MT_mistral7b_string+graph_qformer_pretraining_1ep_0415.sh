export TOKENIZERS_PARALLELISM=false;
file_name='MT_mistral7b_string+graph_qformer_pretraining_1ep_0415'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
gnn=gine_tokengt \
trainer=mistral7b_80gb_llava_pretraining \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false

