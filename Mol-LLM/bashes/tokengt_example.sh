export TOKENIZERS_PARALLELISM=false;
file_name='tokengt_example'
gpus="'4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=ft \
data.data_tag=forward_reaction_prediction_0219 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=TokenGT \
gnn.tune_gnn=False \
trainer=mistral7b_80gb \
trainer.max_epochs=12 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
trainer.total_batch_size=88 \
trainer.every_n_epochs=0