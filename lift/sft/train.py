"""
Edited from https://github.com/GraphPKU/PiSSA
"""
import datasets
import logging
import os
import random
import torch
import torch.distributed
import transformers

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Dict
from .args import TrainingArguments
from .trainer import LIFTSFTTrainer
from .utils import get_last_checkpoint, safe_save_model_for_hf_trainer

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def build_model(script_args: TrainingArguments, checkpoint_dir):
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if not script_args.full_finetune:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize LoRA/PiSSA/CLOVER adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder=script_args.adapter_name_or_path, is_trainable=True)
            if script_args.bf16:
                for name, param in model.named_parameters():
                    if 'lora' in name:
                        param.data = param.data.to(torch.bfloat16)
                        if param.grad is not None:
                            param.grad = param.grad.to(torch.bfloat16)
        else:
            logger.info(f'Init LoRA/PiSSA modules...')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
            )
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.enable_input_require_grads()
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    return model


def train(script_args: TrainingArguments, data_modules: Dict):
    # Set up logging
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Print the arguments
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    # Load tokenizer, keep the same as LIFT
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    # Build model
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir, logger)
    model = build_model(script_args, resume_from_checkpoint_dir)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Distributed training
    # if script_args.local_rank > 0: 
    #     torch.distributed.barrier()

    # Print the dataset
    if script_args.local_rank == 0:
        # torch.distributed.barrier()
        print(model)
        logger.info("Training dataset samples:", len(data_modules['train_dataset']))
        for index in random.sample(range(len(data_modules['train_dataset'])), 3):
            logger.info(f"Sample {index} of the training set: {data_modules['train_dataset'][index]['input_ids']}, {data_modules['train_dataset'][index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(data_modules['train_dataset'][index]['input_ids']))}.")

    # Training
    script_args.accelerator_config.even_batches = False  # due to manually assign the batches
    trainer = LIFTSFTTrainer(model=model, processing_class=tokenizer, args=script_args, **data_modules)
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    
    # Save the model
    trainer.save_state()
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
