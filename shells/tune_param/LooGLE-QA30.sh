#!/bin/bash
#SBATCH -J LooGLE
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-LooGLE-QA30.out.log
#SBATCH -e logs/%j-LooGLE-QA30.err.log
#SBATCH -c 1

python scripts/test_loogle.py \
    --input_file datasets/loogle/longdep_qa.jsonl \
    --output_file outputs/LooGLE-QA30.jsonl \
    --overwrite True \
    --num_syn_qa 30 \
    --title_option 1 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --model_max_length 7900 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --use_lora False \
    --gather_batches True \
    --involve_qa_epochs 3 \
    --num_train_epochs 2 \
    --remove_unused_columns True \
    --report_to none \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --lr_scheduler_type constant

