import json
import numpy as np
import os
import pickle
import torch
import tqdm
import transformers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict, Sequence, Tuple


BAMBOO_FORMAT = "Given a long text, and {num_events} events which take place in the long text, each indicated by number identifier [] that represents the shuffled order, (e.g. [0], [2], etc.). Reorder the events according to the original order of the events in the long text. The events should be listed in descending order using identifiers, and the first event in original order should be list first, and the output format should be [] > [], e.g., [0] > [2]. Only response the reorder results, do not say any word or explain.\n\nLong text:\n{content}\nEvents: {events}\n\nGive the reorder results, only give the ordered identifiers of the {num_events} events {answer_format}: "


class LIFTSFTDataset(Dataset, ABC):
    """Dataset for pre-LIFT SFT."""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, model_max_length: int, num_article_epochs: int, num_article_qa_epochs: int, cache_path: Optional[str]=None, **kwargs):
        super().__init__()
        self.model_max_length = model_max_length
        self.num_article_epochs = num_article_epochs
        self.num_article_qa_epochs = num_article_qa_epochs
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.data = checkpoint['data']
            self.batch_ids = checkpoint['batch_st']
        else:
            with open(data_path, 'r') as f:
                if os.path.splitext(data_path)[1] == '.json':
                    samples = json.load(f)
                elif os.path.splitext(data_path)[1] == '.jsonl':
                    samples = list(map(json.loads, f.readlines()))
            self.data = []
            self.batch_ids = []
            total_datapoints = 0
            for sample in tqdm.tqdm(samples, desc='Processing samples'):
                article_datapoints, qa_datapoints = self.process_single_datapoint(sample, tokenizer, model_max_length=model_max_length, **kwargs)
                self.batch_ids.append((
                    (total_datapoints, total_datapoints + len(article_datapoints)),
                    (total_datapoints, total_datapoints + len(article_datapoints) + len(qa_datapoints))
                ))
                total_datapoints += len(article_datapoints) + len(qa_datapoints)
                self.data += article_datapoints + qa_datapoints
            if cache_path is not None:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'data': self.data,
                        'batch_st': self.batch_ids
                    }, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print('?', i, self.sample_ids[i])
        return self.data[i]
    
    @abstractmethod
    def process_single_datapoint(self, sample, tokenizer: PreTrainedTokenizer, **kwargs) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a single sample into datapoints. It returns two batches of datapoints, the first is the article segments and the second is the QA pairs.
        A datapoint is a dictionary containing the following
        - input_ids (List[int]): tokenized input
        - labels (List[int]): tokenized labels
        """
        raise NotImplementedError


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int = -100

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple([instance[key] for instance in instances] for key in ('input_ids', 'attention_mask', 'labels'))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
