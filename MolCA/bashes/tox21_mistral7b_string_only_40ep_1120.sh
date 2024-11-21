export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=tox21_mistral7b_string_only_40ep_1120 \
data=tox21 \
trainer=mistral7b_lorar32_80gb \
++trainer.mol_representation=string_only \
++data.raw_data_root="'MolCA/data/multi_task_0927'" \
++trainer.selfies_token_path="'MolCA/model/selfies_dict.txt'" \
++trainer.logging_dir="'MolCA/all_checkpoints/'" \

