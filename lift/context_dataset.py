import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from typing import List, Tuple, Optional
from copy import deepcopy
from nltk import sent_tokenize
from tqdm import tqdm
from random import randint


def apply_qa_template(question: str, answer: Optional[str]=None, evidences: Optional[List[str]]=None, title: Optional[str]=None, context: Optional[str]=None, return_answer: bool=False):
    """Apply the QA template, used for training.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer; it should be equipped with a chat template.
        question (str):
        answer (str):
        evidences (list[str]): the evidences; it should be presented as a list of sentences.
        title (str):
        prepend_title (bool): OPTIONAL; whether to prompt the model with the title.
        sent_token (str|None): if specified as non-None value, a `sent_token` will be prepended to each sentence in the evidences.
    Returns:
        PAIR (tuple[Tensor, int]): input_ids - the `input_ids` of the model; len_input - the length of the unsupervised texts, including the system prompt, the context, and the question.
    """
    prompts = []
    if title is not None:
        prompts.append(f"Please answer the following question only based on the article \"{title}\".")
        if context is not None:
            prompts.append(f"This is part of the article \"{title}\": \n{context}\n")
        if evidences is not None:
            prompts.append(f"Please recite the facts from \"{title}\" that support your answer before answering the question according to the facts.")
    else:
        prompts.append("Please answer the following question.")
        if context is not None:
            prompts.append(f"This is part of the texts: \"{context}\"")
        if evidences is not None:
            prompts.append(f"Please recite the facts from the text that support your answer before answering the question according to the facts.")
    prompts.append(f"Question:\n{question}\n")
    if evidences is not None:
        prompts.append("Please answer in the following format: \"Evidence: <facts>. Answer: <answer>\". Do NOT output anything else.")
    else:
        prompts.append("Please answer in the following format: \"Answer: <answer>\". Do NOT output anything else.")
    if return_answer:
        if evidences is not None:
            output = f"Evidence: {' '.join(evidences)} Answer: {answer}"
        else:
            output = f"Answer: {answer}"
        return '\n'.join(prompts), output
    else:
        return '\n'.join(prompts)


class ContextDataset(Dataset):
    """Given a piece of context, `ContextDataset` creates a torch-Dataset, using the truncation strategy described in our paper.
    """
    def __init__(self, context: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3):
        """
        Args:
            context (str): the context to train on.
            tokenizer (PreTrainedTokenizer): the AutoTokenizer.
            model_max_length (int): OPTIONAL, default to `4096`; the texts will be clipped at the `model_max_length`-th token.
            block_size (int): OPTIONAL, default to `256`; the number of tokens in a block; a block is the unit of segments and offsets.
            len_segment (int): OPTIONAL, default to `8`; the number of units in a segment; the article is divided into segments.
            len_offset (int): OPTIONAL, default to `3`; the number of units per offset; it determines the offset from one segment to the next one.
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

    def __len__(self):
        return self.num_segments

    def preprocessing(self, example: Tuple[List[int], int]):
        input_ids, len_input = example
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
        }
    
    def __getitem__(self, index):
        return self.preprocessing(self.data[index])
    
    def enable_qa(self):
        raise NotImplementedError
    
    def disable_qa(self):
        raise NotImplementedError
    
    def generate_task(self):
        raise NotImplementedError
