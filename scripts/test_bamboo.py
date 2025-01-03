#!/usr/bin/env python
# coding=utf-8
import logging
import tqdm
import json
import torch
import os
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from dataclasses import dataclass, field
from typing import Optional
from lift.args import ModelArguments, CustomTrainingArguments, DataTrainingArguments, parse_args
from lift.train import train
from lift.context_dataset import ContextDataset
from lift.model import load_tokenizer, load_model
from nltk import sent_tokenize
from numpy.random import choice


BAMBOO_PROMPT_FORMAT = "Given a long text, and {num_events} events which take place in the long text, each indicated by number identifier [] that represents the shuffled order, (e.g. [0], [2], etc.). Reorder the events according to the original order of the events in the long text. The events should be listed in descending order using identifiers, and the first event in original order should be list first, and the output format should be [] > [], e.g., [0] > [2]. Only response the reorder results, do not say any word or explain.\n\nLong text:\n{content}\nEvents: {events}\n\nGive the reorder results, only give the ordered identifiers of the {num_events} events {answer_format}: "


@dataclass
class TestArguments:
    input_file: str = field(metadata={"help": "The input file for the test."})
    output_file: Optional[str] = field(default=None, metadata={"help": "The output file for the test."})
    overwrite: bool = field(default=False, metadata={"help": "Whether to overwrite the output file."})
    num_syn_qa: int = field(default=0, metadata={"help": "The number of synthetic QA pairs to generate."})


class BambooDataset(ContextDataset):
    def __init__(self, context: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, num_syn_qa: int=0):
        super().__init__(context, tokenizer, model_max_length, block_size, len_segment, len_offset)
        for _ in range(num_syn_qa):
            self.data.append(self.generate_task(context, model_max_length))
        self.enable_qa_tag = False
    
    def generate_task(self, full_context: str, model_max_length: int=4096):
        """Synthesize a 5-event timeline reorder task.
        Args:
            full_context: The full context for the timeline reorder task.
            model_max_length: The maximum length of the model input.
        """
        sentences = [s for s in sent_tokenize(full_context) if len(s) >= 50]
        assert len(sentences) >= 5, "No enough long sentences to generate sentence-based timeline reorder task."
        sentence_ids = choice(len(sentences), 5, replace=False).tolist()
        summaries = [sentences[i] for i in sentence_ids]
        answers = [0, 1, 2, 3, 4]
        answers.sort(key=lambda i: sentence_ids[i])
        prompt = BAMBOO_PROMPT_FORMAT.format_map(dict(
            num_events=5,
            content=full_context,
            events=summaries,
            answer_format=' < '.join(['[]'] * 5)
        ))
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': ' < '.join([f"[{i + 1}]" for i in answers])}
        ]
        input_length = len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        output_length = len(input_ids) - input_length
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2] + input_ids[-model_max_length//2:]
            input_length = len(input_ids) - output_length
        return (input_ids, input_length)
    
    def enable_qa(self):
        self.enable_qa_tag = True
    
    def disable_qa(self):
        self.enable_qa_tag = False
    
    def __len__(self):
        return len(self.data) if self.enable_qa_tag else self.num_segments


def BambooTrain(model_name_or_path: str, context: str, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, num_syn_qa: int=0, involve_qa_epochs: int=0, use_lora: bool=False, lora_rank: Optional[int]=None, use_pissa: bool=False, load_in_4bit: bool=False, gather_batches: bool=True, **kwargs):
    dataset = BambooDataset(context, tokenizer, model_max_length, block_size, len_segment, len_offset, num_syn_qa)
    model = load_model(model_name_or_path=model_name_or_path, use_lora=use_lora, lora_rank=lora_rank, use_pissa=use_pissa, load_in_4bit=load_in_4bit, vocab_size=len(tokenizer))
    model = train(model, dataset, tokenizer, training_args, involve_qa_epochs, gather_batches)[0]
    return model


def prediction(dataset: list[dict], training_args: TrainingArguments, lift_args: dict, output_file: str, num_resumed: int=0, num_syn_qa: int=0, **kwargs):
    logging.warning("Unused arguments: %s", kwargs)
    model_max_length = lift_args['model_max_length']
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    for index, sample in enumerate(tqdm.tqdm(dataset, total=len(dataset), desc="Predicting")):
        if index < num_resumed:
            continue
        summaries = sample["summaries"]
        prompt = BAMBOO_PROMPT_FORMAT.format_map({
            'num_events': len(summaries),
            'events': '\n'.join(f"[{i + 1}]: {summaries[i]}" for i in range(len(summaries))),
            'content': sample['content'],
            'answer_format': ' < '.join(['[]'] * len(summaries))
        })
        model = BambooTrain(context=sample['content'], tokenizer=tokenizer, training_args=training_args, num_syn_qa=num_syn_qa, **lift_args)
        torch.cuda.empty_cache()
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2] + input_ids[-model_max_length//2:]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)[None, :]
        mask_attention = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        output = model.generate(
            input_ids=input_ids,
            attention_mask=mask_attention,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
            max_new_tokens=32,
            temperature=0.7,
            use_cache=False,
        )
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        sample['pred'] = response
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")

def main():
    training_args, test_args, args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    input_file = test_args.pop('input_file')
    overwrite = test_args.pop('overwrite')
    output_file = test_args['output_file']
    if not overwrite and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            num_resumed = len(f.readlines())
    else:
        if os.path.exists(test_args['output_file']):
            os.remove(test_args['output_file'])
        num_resumed = 0
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f.readlines()]
    prediction(dataset, training_args, args, num_resumed=num_resumed, **test_args)

if __name__ == "__main__":
    main()
