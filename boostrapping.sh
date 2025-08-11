#!/bin/bash

GPU=0
# Number of rounds for bootstrapping
num_rounds=5
# Number of samples
sample_num=8

train_list=$1
train_image=$2

# Create a root output directory
root_output_dir="./exp"
mkdir -p $root_output_dir

# Optionally, you can pass an input file name as the first argument
input_file="$1"

if [[ -n "$input_file" ]]; then
    unique_dir="$root_output_dir/$input_file"
    mkdir -p $unique_dir
else
    unique_dir=$(mktemp -d "$root_output_dir/unique_run.XXXXXX")
fi

echo "All files will be stored in: $unique_dir"

for ((j=0; j<${#train_list[@]}; j++)); do
    train_file=${train_list[j]}
    image_folder=${train_image[j]}

    python tools/get_sub_question.py \
        --file_name ${unique_dir} \
        --train_file  ${train_file} \
        --train_image ${image_folder} \
        --sample_num ${sample_num}

    for ((i=0; i<num_rounds; i++)); do
        if [ $i -eq 0 ]; then
            # First round, use the original model
            echo "***Round $i***"
            echo "***Inference the box***"
            CUDA_VISIBLE_DEVICES=${GPU} python -m llava.eval.model_sub_question \
                --model-path deepcs233/VisCoT-7b-336 \
                --question-file ${unique_dir}/${train_file}_splited.json \
                --image-folder ${image_folder} \
                --answers-file $unique_dir/train_box_round$i.jsonl \
                --temperature 0 \
                --conv-mode vicuna_v1 

            echo "***Finish box***"
            wait

            echo "***Checking the box***"
            CUDA_VISIBLE_DEVICES=${GPU} python -m llava.eval.eval_check_box \
                --model-path deepcs233/VisCoT-7b-336 \
                --question-file $unique_dir/train_box_round$i.jsonl \
                --image-folder ${image_folder} \
                --answers-file $unique_dir/train_check_round$i.jsonl \
                --temperature 0 \
                --conv-mode vicuna_v1 \
                --repeat 1

        else
            # Subsequent rounds, use the model from the previous round
            echo "***Round $i***"
            echo "***Inference the box***"
            CUDA_VISIBLE_DEVICES=${GPU} python -m llava.eval.model_sub_question \
                --model-base deepcs233/VisCoT-7b-336 \
                --model-path $unique_dir/llava_lora_round$i \
                --question-file ${unique_dir}/${train_file}_splited.json \
                --image-folder ${image_folder} \
                --answers-file $unique_dir/train_box_round$i.jsonl \
                --temperature 0 \
                --conv-mode vicuna_v1 

            echo "***Finish box***"
            wait

            echo "***Checking the box***"
            CUDA_VISIBLE_DEVICES=${GPU} python -m llava.eval.eval_check_box \
                --model-base deepcs233/VisCoT-7b-336 \
                --model-path $unique_dir/llava_lora_round$i \
                --question-file $unique_dir/train_box_round$i.jsonl \
                --image-folder ${image_folder} \
                --answers-file $unique_dir/train_check_round$i.jsonl \
                --temperature 0 \
                --conv-mode vicuna_v1 \
                --repeat 1

            echo "***Finish check***"

        fi

        python tools/box_check.py \
            --check_file $unique_dir/train_check_round$i.jsonl \
            --save_path $unique_dir/checked_round$i.json \
            --base_path $unique_dir/ \
            --sampled_file ${unique_dir}/${train_file}_sampled_${sample_num}.json

        echo "***Saved the correct sample, Start training***"

        deepspeed --include localhost:${GPU} llava/train/train_mem.py \
            --lora_enable True --lora_r 32 --lora_alpha 64 \
            --deepspeed ./scripts/zero3.json \
            --model_name_or_path deepcs233/VisCoT-7b-336 \
            --freeze_backbone True \
            --freeze_mm_mlp_adapter True \
            --version v1 \
            --data_path $unique_dir/checked_round$i.json \
            --image_folder ${image_folder} \
            --vision_tower openai/clip-vit-large-patch14-336 \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length False \
            --bf16 True \
            --output_dir $unique_dir/llava_lora_round$(expr $i + 1) \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "epoch" \
            --save_total_limit 1 \
            --learning_rate 2e-4 \
            --weight_decay 0. \
            --warmup_ratio 0.01 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to none

        echo "Generated files for round $i stored in: $unique_dir"

    done

    echo "***Inference the box***"
    CUDA_VISIBLE_DEVICES=${GPU} python -m llava.eval.model_sub_question \
        --model-base deepcs233/VisCoT-7b-336 \
        --model-path $unique_dir/$task/llava_lora_round$i \
        --question-file ${unique_dir}/${train_file}_splited.json \
        --image-folder ${image_folder} \
        --answers-file $unique_dir/train_box_round$i.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 

    echo "***Checking the box***"
    CUDA_VISIBLE_DEVICES=${GPU} python -m llava.eval.eval_check_box \
        --model-base deepcs233/VisCoT-7b-336 \
        --model-path $unique_dir/$task/llava_lora_round$i \
        --question-file $unique_dir/train_box_round$i.jsonl \
        --image-folder ${image_folder} \
        --answers-file $unique_dir/train_check_round$i.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --repeat 1
    echo "***Finish check***"

    python tools/box_check.py \
    --check_file $unique_dir/train_check_round$i.jsonl \
    --save_path $unique_dir/checked_round$i.json \
    --base_path $unique_dir/ \

    echo "All rounds completed. All files are stored in: $unique_dir"

    python tools/get_gcot.py \
        --file_name $unique_dir \
        --num_rounds $num_rounds \
        --train_file $train_file \
done