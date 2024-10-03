export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_llama3.1-8b_string_only_1003 \
data=multi_task \
trainer=8b_80gb \
llm=llama3.1-8b-instruct
