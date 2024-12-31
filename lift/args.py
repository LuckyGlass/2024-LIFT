from dataclasses import dataclass, field
from typing import Optional, Any, List
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_max_length: Optional[int] = field(default=None)
    is_peft_model: bool = field(default=False)  # TODO: auto detect PEFT models
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    len_segment: int = field(
        default=2,
        metadata={
            "help": (
                "The number of blocks in a segment."
            )
        }
    )
    len_offset: int = field(
        default=1,
        metadata={
            "help": (
                "The offset from one segment to the next segment."
            )
        }
    )


@dataclass
class CustomTrainingArguments:
    use_lora: bool = field(default=False)
    lora_rank: int = field(default=8)
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)
    gather_batches: bool = field(default=False)
    involve_qa_epochs: int = field(default=0)


def parse_args(class_clusters: tuple[Any|tuple[Any]], no_dict: tuple[Any], return_config: bool=False):
    class_set = set()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            class_set.update(set(cluster))
        else:
            class_set.add(cluster)
    class_tuple = tuple(class_set)
    parser = HfArgumentParser(class_tuple)
    arg_list = parser.parse_args_into_dataclasses()
    arg_dict = {c: a for c, a in zip(class_tuple, arg_list)}
    returns = ()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            temp = {}
            for item in cluster:
                temp.update(dict(vars(arg_dict[item])))
            returns += (temp,)
        else:
            if cluster in no_dict:
                returns += (arg_dict[cluster],)
            else:
                returns += (dict(vars(arg_dict[cluster])),)
    if return_config:
        config = {}
        for arg in arg_list:
            config.update({k: v for k, v in dict(vars(arg)).items() if isinstance(v, int|float|bool|str)})
        return returns, config
    return returns
