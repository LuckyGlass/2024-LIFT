import torch
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
    PreTrainedModel
)
from .model import load_model
from .context_dataset import ContextDataset, DatasetWithSyntheticQA
from typing import Optional, Type
from copy import deepcopy


def load_trainer(model: PreTrainedModel, training_dataset: Dataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, eval_dataset: Optional[Dataset]=None, gather_batches: bool=False, optimizer: Optional[torch.optim.Optimizer]=None):
    """Load the training and the model (if the model is not instantiated).
    Args:
        training_dataset (Dataset):
        tokenizer (PreTrainedTokenizer):
        training_args (TrainingArguments):
        eval_dataset (Dataset): OPTIONAL, default to `None`.
        model_name_or_path (str): OPTIONAL, default to `None`; if the model is not instantiated, assign it to load the model.
        gather_batches (bool): OPTIONAL, default to `False`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
        model (Module): OPTIONAL, default to `None`; a model instance.
        optimizer (Optimizer): OPTIONAL, default to `NONE`; an optimizer instance.
        use_lora (bool): OPTIONAL, default to `False`; whether to use LoRA.
        lora_rank (int): OPTIONAL, default to `None`; assign it when `use_lora=True`.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
    Returns:
        trainer_model_pair (tuple[Trainer, Module]): the trainer and the model to train.
    """
    training_args = deepcopy(training_args)
    if gather_batches:
        training_args.gradient_accumulation_steps = len(training_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None)
    )
    return trainer, model


def train(model: PreTrainedModel, dataset: ContextDataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, involve_qa_epochs: int=0, gather_batches: bool=True):
    """Fine-tune the model and the corresponding tokenizer.
    Args:
        dataset (Dataset): the dataset to train on.
        tokenizer (PreTrainedTokenizer): a Llama tokenizer (or other tokenizers with chat template).
        training_args (TrainingArguments): transformers-style training arguments, used for the trainer.
        gather_batches (bool): OPTIONAL, default to `False`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
        use_lora (bool): OPTIONAL, default to `False`; whether to use LoRA.
        lora_rank (int): OPTIONAL, default to `None`; assign it when `use_lora=True`.
        full_ft (bool): OPTIONAL, default to `False`; whether to full-fine-tune the model.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        cache_dir (str): OPTIONAL, default to `None`.
        model_revision (str): OPTIONAL, default to `"main"`.
        use_auth_token (bool): OPTIONAL, default to `False`.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """
    # load tokenzier
    torch.cuda.empty_cache()  # Manually release memory
    # Load and finetune the model
    if isinstance(dataset, DatasetWithSyntheticQA):
        dataset.disable_qa()
    trainer, model = load_trainer(
        model=model,
        training_dataset=dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        gather_batches=gather_batches,
    )
    if training_args.num_train_epochs > 0:
        trainer.train()
    # Load the dataset with QA pairs and continue-finetune the model
    if involve_qa_epochs > 0:
        if isinstance(dataset, DatasetWithSyntheticQA):
            dataset.enable_qa()
        training_args_syn = deepcopy(training_args)
        training_args_syn.num_train_epochs = involve_qa_epochs
        trainer_syn, model = load_trainer(
            model=model,
            training_dataset=dataset,
            tokenizer=tokenizer,
            training_args=training_args_syn,
            gather_batches=gather_batches,
            optimizer=trainer.optimizer,
        )
        trainer_syn.train()
    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
