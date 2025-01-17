#!/bin/bash
#SBATCH -J NSent
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:1
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-LooGLE-ShortQA-NSent-30.out.log
#SBATCH -e logs/%j-LooGLE-ShortQA-NSent-30.err.log
#SBATCH -c 1

python scripts/test_loogle.py \
    --input_file datasets/loogle/shortdep_qa.jsonl \
    --output_file outputs/LooGLE-ShortQA-NSent-30.jsonl \
    --overwrite True \
    --num_syn_qa 30 \
    --title_option 1 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --use_cot True \
    --num_test 50 \
    --model_name_or_path models/Llama-3-8B-Instruct-pissa-r128 \
    --model_max_length 7800 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --use_lora True \
    --lora_rank 128 \
    --use_pissa True \
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

