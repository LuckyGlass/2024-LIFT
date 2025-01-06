#!/bin/bash
#SBATCH -J LIFTSFT
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-LIFTSFT.out.log
#SBATCH -e logs/%j-LIFTSFT.err.log
#SBATCH -c 1

deepspeed --master_port=16971 scripts/sft_train.py \
    --data_path data/sft_train.jsonl \
    --len_segment 8 \
    --len_offset 3 \
    --block_size 256 \
    --input_cache_path cache/sft_train.pkl \
    --num_article_epochs 2 \
    --num_article_qa_epochs 3 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --num_syn_qa 20 \
    --deepspeed shells/ds_config_zero2_no_offload.json \
    --model_name_or_path models/Llama-3-8B-Instruct-pissa-r128 \
    --full_finetune False \
    --bf16 \
    --adapter_name_or_path "pissa_init" \
    --output_dir models/sft \
    --num_train_epochs 5 \
    --model_max_length 8000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --merge True \
    --gradient_checkpointing True
