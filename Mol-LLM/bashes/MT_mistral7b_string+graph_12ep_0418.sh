export TOKENIZERS_PARALLELISM=false;
file_name=MT_mistral7b_string+graph_${gnn}_12ep_0418
gpus=$1
gnn=$2

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
data.data_tag=ablation_0411 \
gnn=${gnn} \
trainer=mistral7b_80gb \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false

