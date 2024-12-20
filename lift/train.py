import torch
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer
)
from .model import load_model, load_optimizer
from .context_dataset import ContextDataset, DatasetWithSyntheticQA
from typing import Optional, Type
from copy import deepcopy


def load_trainer(training_dataset: Dataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, eval_dataset: Optional[Dataset]=None, model_name_or_path: Optional[str]=None, gather_batches: bool=False, model: Optional[torch.nn.Module]=None, optimizer: Optional[torch.optim.Optimizer]=None, use_lora: bool=False, lora_rank: Optional[int]=None, load_in_4bit: bool=False, load_in_8bit: bool=False, vocab_size: Optional[int]=None):
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
    # If the model is not given, then load the model
    if model is None:
        model = load_model(model_name_or_path, use_lora, lora_rank, load_in_4bit, load_in_8bit, vocab_size)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None)
    )
    return trainer, model


def train(dataset: ContextDataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, involve_qa_epochs: int=0, **kwargs):
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
    dataset.disable_qa()
    trainer, model = load_trainer(dataset, tokenizer, training_args, trainer_cls=Trainer, **kwargs)
    if training_args.num_train_epochs > 0:
        trainer.train()
    # Load the dataset with QA pairs and continue-finetune the model
    if involve_qa_epochs > 0:
        dataset.enable_qa()
        training_args_syn = deepcopy(training_args)
        training_args_syn.num_train_epochs = involve_qa_epochs
        trainer_syn, model = load_trainer(dataset, tokenizer, training_args_syn, model=model, optimizer=trainer.optimizer, trainer_cls=Trainer, **kwargs)
        trainer_syn.train()
    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
