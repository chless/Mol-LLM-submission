export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3'

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--opt_model facebook/galactica-1.3b \
--llm_tune lora \
--lora_r 64 \
--per_device_batch_size_cls 3 \
--per_device_batch_size_reg 15 \
--per_device_batch_size_rxn 10 \
--per_device_batch_size_rea 5 \
--per_device_batch_size_trn 8 \
--per_device_inference_batch_size_cls 3 \
--per_device_inference_batch_size_reg 15 \
--per_device_inference_batch_size_rxn 10 \
--per_device_inference_batch_size_rea 5 \
--per_device_inference_batch_size_trn 8 \
--val_check_interval 2500 \
--max_epochs 15 \
--second_stage_start_epoch 5 \
--num_beam 1 \
--raw_data_root /data/datasets/multi_task_dataset_0813_selfies \
--gen_max_len 300 \
--prompt_max_len 300 \
--label_max_len 300 \
--logging_dir /data/text-mol-logging \
--mol_representation string+graph \
--filename MT_galac1.3b_lorar64_llava1_string+graph_no_random_selfies_0814 \
--graph_encoder_ckpt /data/ckpts//MoleculeSTM/molecule_model.pth \
--resize 2400 \
--skip_sanity_check \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--accumulate_grad_batches 2