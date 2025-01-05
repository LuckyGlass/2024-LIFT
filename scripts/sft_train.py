from lift.sft.args import LIFTDataArguments, TrainingArguments
from lift.sft.dataset import LIFTSFTDataset, DataCollatorForSupervisedDataset
from lift.sft.train import train
from nltk import sent_tokenize
from random import randint
from transformers import PreTrainedTokenizer, HfArgumentParser, AutoModelForCausalLM, PreTrainedModel, BitsAndBytesConfig, AutoTokenizer
from typing import Dict, List
import logging
import torch
import tqdm


LOOGLEFORMAT = "The article {title}: \n{input}\nPlease answer the question based on {title}.\nQuestion: {question}\nAnswer: "


class GeneralTaskSFTDataset(LIFTSFTDataset):
    def process_single_datapoint(self, sample, tokenizer: PreTrainedTokenizer, len_segment: int, len_offset: int, block_size: int, num_syn_qa: int, generator: PreTrainedModel, model_max_length: int, **kwargs):
        """
        Process a single sample into datapoints. It returns two batches of datapoints, the first is the article segments and the second is the QA pairs.
        A datapoint is a dictionary containing the following
        - input_ids (Tensor): tokenized input
        - attention_mask (Tensor): attention masks
        - labels (Tensor): tokenized labels
        """
        title = sample['title']
        context = sample['input']
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size
        snippet = tokenizer(f"A snippet of {title}: ", add_special_tokens=False)['input_ids']
        
        article_tokens = tokenizer(context.replace('\0', ' '), add_special_tokens=False)['input_ids']
        # Generate article datapoints
        article_batches = []
        for s in range(0, len(article_tokens), len_offset):
            input_ids = torch.tensor(snippet + article_tokens[s:s+len_segment], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            attention_mask[:len(snippet)] = 0
            labels[:len(snippet)] = -100
            article_batches.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        # Generate QA datapoints
        context_sent = sent_tokenize(context)
        assert len(context_sent) >= 25, "The length of the context should be at least 25 sentences."
        qa_batches = []
        for _ in tqdm.tqdm(range(num_syn_qa), desc='Generating QA pairs'):
            result = self.generate_task(tokenizer, generator, context, context_sent, title, model_max_length)
            if result is not None:
                qa_batches.append(result)
        return article_batches, qa_batches
    
    @torch.no_grad()
    def generate_task(self, tokenizer: PreTrainedTokenizer, generator: PreTrainedModel, full_context: str, context_sent: List[str], title: str, model_max_length: int):
        st_pos = randint(0, len(context_sent) - 25)
        context = ' '.join(context_sent[st_pos:st_pos+25])
        messages = [
            {
                'role': "system",
                'content': "You are a helpful assistant."
            },
            {
                'role': "user", 
                'content': f"You are given a piece of text as the context. You should generate ONLY one question and the corresponding answer according to the context. You should also select one or more original sentences in the context as the evidences. Please answer in the following format:\nQuestion: [question]\nAnswer: [answer]\nEvidence:\n- [evidence 1]\n- [evidence 2]\n...\nPlease DON'T output quotes when outputing evidences. The following is the piece of text: {context}"
            }
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(generator.device)
        mask_attention = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        for _ in range(5):
            outputs = generator.generate(
                input_ids=input_ids,
                attention_mask=mask_attention.to(generator.device),
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=False,
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            question_position = response.find("Question:")
            answer_position = response.find("Answer:")
            evidence_position = response.find("Evidence:")
            if question_position == -1 or answer_position == -1 or evidence_position == -1:
                continue
            question = response[question_position + 9:answer_position].strip()
            answer = response[answer_position + 7:evidence_position].strip()
            evidences = response[evidence_position + 9:].strip().split('\n')
            evidences = list(map(lambda s: s[s.find('-') + 2:].strip(), evidences))
            break
        else:
            logging.warning("Failed to generate a QA pair.")
            return None
        input_text = LOOGLEFORMAT.format(title=title, input=full_context, question=question)
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': input_text},
            {'role': 'assistant', 'content': "Evidences:\n" + '\n'.join([f"- {e}" for e in evidences]) + "\nAnswer:\n" + answer}
        ]
        input_length = len(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        mixin = tokenizer("...", add_special_tokens=False)['input_ids']
        output_length = len(input_ids) - input_length
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_length = len(input_ids) - output_length
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:input_length] = 0
        labels = input_ids.clone()
        labels[:input_length] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, model_max_length: int, num_article_epochs: int, num_article_qa_epochs: int, cache_path: str, len_segment: int, len_offset: int, block_size: int, num_syn_qa: int, generator_name_or_path: str):
        generator = AutoModelForCausalLM.from_pretrained(
            generator_name_or_path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
        generator.eval()
        super().__init__(data_path, tokenizer, model_max_length, num_article_epochs, num_article_qa_epochs, cache_path, len_segment=len_segment, len_offset=len_offset, block_size=block_size, num_syn_qa=num_syn_qa, generator=generator)



def load_lift_dataset(tokenizer: PreTrainedTokenizer, data_args: LIFTDataArguments, model_max_length: int) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = GeneralTaskSFTDataset(
        data_args.data_path,
        tokenizer,
        model_max_length,
        data_args.num_article_epochs,
        data_args.num_article_qa_epochs,
        data_args.input_cache_path,
        data_args.len_segment,
        data_args.len_offset,
        data_args.block_size,
        data_args.num_syn_qa,
        data_args.generator_name_or_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, LIFTDataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    data_modules = load_lift_dataset(tokenizer, data_args, training_args.model_max_length)
    train(training_args, data_modules)
