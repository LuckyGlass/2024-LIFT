import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from typing import List, Tuple, Union
from copy import deepcopy


class ContextDataset(Dataset):
    """Given a piece of context, `ContextDataset` creates a torch-Dataset, using the truncation strategy described in our paper.
    """
    def __init__(self, context: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, regularization_scale: float=.0):
        """
        Args:
            context (str): the context to train on.
            tokenizer (PreTrainedTokenizer): the AutoTokenizer.
            model_max_length (int): OPTIONAL, default to `4096`; the texts will be clipped at the `model_max_length`-th token.
            block_size (int): OPTIONAL, default to `256`; the number of tokens in a block; a block is the unit of segments and offsets.
            len_segment (int): OPTIONAL, default to `8`; the number of units in a segment; the article is divided into segments.
            len_offset (int): OPTIONAL, default to `3`; the number of units per offset; it determines the offset from one segment to the next one.
            regularization_scale (float): OPTIONAL, default to `0`; the memgate regularization scale.
        """
        self.ignore_index = -100  # The default value for ignored labels in torch
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        texts = context.replace('\0', ' ')
        input_ids = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size
        # Generate datapoints
        self.data = [(input_ids[s:s+len_segment], 0) for s in range(0, len(input_ids), len_offset)]
        self.num_segments = len(self.data)  # record the number of context datapoints
        self.regularization_scale = regularization_scale

    def __len__(self):
        return self.num_segments

    def preprocessing(self, example: Union[Tuple[List[int], int], Tuple[List[int], int, bool]]):
        if len(example) == 3:
            input_ids, len_input, do_regularization = example
        else:
            input_ids, len_input = example
            do_regularization = True
        labels = deepcopy(input_ids)
        # Clip and truncation
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        # Transfer to Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[:len_input] = self.ignore_index  # mask the unsupervised part
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'regularization_scale': self.regularization_scale if do_regularization else .0
        }
    
    def __getitem__(self, index):
        return self.preprocessing(self.data[index])
    
    def enable_qa(self):
        raise NotImplementedError
    
    def disable_qa(self):
        raise NotImplementedError
    
    def generate_task(self):
        raise NotImplementedError
