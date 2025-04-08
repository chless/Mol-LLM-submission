export TOKENIZERS_PARALLELISM=false;
gpus=$1
projector_type=$2
modality=$3
task=$4

list_iso=(
    0
    1
    )
list_can=(
    0
    1
    )
list_Hs=(
    0
    1
    )


for iso in "${list_iso[@]}"; do
    for can in "${list_can[@]}"; do
        for Hs in "${list_Hs[@]}"; do
            executing_filename=${task}_${projector_type}_${modality}_12ep_iso${iso}_can${can}_Hs${Hs}_0318
            echo "==============Executing filename: $executing_filename==============="
            python Mol-LLM/stage3.py \
            trainer.devices=$gpus \
            mode=test \
            filename=${executing_filename} \
            data.data_tag=${task}_0219 \
            data.raw_data_root=/data/data/Mol-LLM-v7.1 \
            gnn=moleculeSTM \
            gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
            trainer=mistral7b_80gb \
            trainer.selfies_enumeration=true \
            trainer.isomericSmiles=${iso} \
            trainer.canonical=${can} \
            trainer.allHsExplicit=${Hs} \
            trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
            trainer.logging_dir=/data/all_checkpoints \
            trainer.mol_representation=${modality} \
            trainer.projector_type=${projector_type} \
            ckpt_path="'/data/all_checkpoints/${task}_${projector_type}_${modality}_12ep_0318/last.ckpt'"
        done
    done
done

