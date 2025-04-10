export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn=TokenGT
task=qm9_homo
projector_type=qformer
max_epochs=12
total_batch_size=88
trained_tokengt_ckpt=/data/all_checkpoints/Custom_gnn_models/TokenGT/best-model.ckpt
moleculeSTM_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth

for modality in graph_only string+graph; do
    for gnn_graph_encoder_ckpt in scratch $trained_tokengt_ckpt; do
        echo "==============Executing task: $task with modality: $modality and gnn_graph_encoder_ckpt: $gnn_graph_encoder_ckpt==============="
        python Mol-LLM/stage3.py \
        trainer.devices=$gpus \
        filename=${task}_${modality}_tokengt_${gnn_graph_encoder_ckpt}_12ep_0409 \
        data.data_tag=${task}_0219 \
        data.raw_data_root=/data/data/Mol-LLM-v7.1 \
        gnn=${gnn} \
        gnn.graph_encoder_ckpt=${gnn_graph_encoder_ckpt} \
        trainer=mistral7b_80gb \
        trainer.max_epochs=${max_epochs} \
        trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
        trainer.logging_dir=/data/all_checkpoints \
        trainer.mol_representation=${modality} \
        trainer.projector_type=${projector_type} \
        trainer.skip_sanity_check=false \
        trainer.total_batch_size=${total_batch_size} \
        trainer.every_n_epochs=0
    done
done

