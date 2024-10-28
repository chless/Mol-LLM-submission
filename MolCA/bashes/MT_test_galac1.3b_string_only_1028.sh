export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_galac1.3b_string_only_1026 \
data=multi_task_extended-v2 \
trainer=galactica1.3b_80gb \
++trainer.mol_representation=string_only \
++mode=test \
++ckpt_path="'/data/ckpts/molllm-ckpt/MT_galac1.3b_string_only_1026/step=17600-train_total_loss=0.469.ckpt'" \
++trainer.logging_dir=/data/text-mol/ \
++trainer.selfies_token_path=MolCA/model/selfies_dict.txt \
++data.raw_data_root=/data/datasets/multi_task_0927/ \
++ckpt_path="'/data/ckpts/molllm-ckpt/MT_mistral7b_string_only_1026/step=05900-train_total_loss=0.380.ckpt'"