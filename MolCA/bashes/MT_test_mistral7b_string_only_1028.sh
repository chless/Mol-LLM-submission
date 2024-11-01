export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_test_mistral7b_string_only_2ep_1028 \
data=multi_task_extended-v2 \
trainer=mistral7b_80gb \
++trainer.mol_representation=string_only \
++mode=test \
++trainer.logging_dir=/data/text-mol/ \
++trainer.selfies_token_path=MolCA/model/selfies_dict.txt \
++data.raw_data_root=/data/datasets/multi_task_0927/ \
++ckpt_path="'/data/ckpts/molllm-ckpt/MT_mistral7b_string_only_1026/step=08900-train_total_loss=0.257.ckpt'"