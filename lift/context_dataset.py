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


class DatasetWithSyntheticQA(ContextDataset):
    def __init__(self, context: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, num_generate_qa: int=0, generator_name_or_path: Optional[str]=None, title: Optional[str]=None):
        super().__init__(context, tokenizer, model_max_length, block_size, len_segment, len_offset)
        # Load the generator
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        generator = AutoModelForCausalLM.from_pretrained(
            generator_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype = torch.bfloat16,
            quantization_config=quantization_config,
        )
        # Generate synthetic QAs
        self.num_generate_qa = num_generate_qa
        temp_qas = self.shortqa_gen(generator, context, num_generate_qa)
        torch.cuda.empty_cache()
        for qa in temp_qas:
            user_msg, assistant_msg = apply_qa_template(
                question=qa['Q'],
                answer=qa['A'],
                evidences=qa['S'],
                title=title,
                context=context,
                return_answer=True
            )
            messages = [
                {'role': 'system', 'content': "You are a helpful assistant."},
                {'role': 'user', 'content': user_msg},
                {'role': 'assistant', 'content': assistant_msg}
            ]
            input_length = len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
            output_length = len(input_ids) - input_length
            if len(input_ids) > model_max_length:
                input_ids = input_ids[:model_max_length//2] + input_ids[-model_max_length//2:]
                input_length = len(input_ids) - output_length
            self.data.append((input_ids, input_length))
        # Set tag
        self.enable_syn_qa = False

    def __len__(self):
        return self.num_segments + self.num_generate_qa if self.enable_syn_qa else self.num_segments

    @torch.no_grad()
    def shortqa_gen(self, generator, full_context: str, num_generate_qa: int=0):
        generated = []
        texts = sent_tokenize(full_context)
        c = 0
        assert len(texts) >= 25
        for _ in tqdm.tqdm(range(num_generate_qa), desc="Generate QA"):
            st_pos = randint(0, len(texts) - 25)
            context = ' '.join(texts[st_pos:st_pos+25])
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
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            mask_attention = torch.ones_like(input_ids)
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            num_of_trials = 0
            while True:
                outputs = generator.generate(
                    input_ids.to(generator.device),
                    max_new_tokens=1024,
                    attention_mask=mask_attention.to(generator.device),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=terminators,
                    do_sample=False,
                )
                response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
                question_position = response.find("Question:")
                answer_position = response.find("Answer:")
                evidence_position = response.find("Evidence:")
                question = response[question_position + 9:answer_position].strip()
                answer = response[answer_position + 7:evidence_position].strip()
                evidences = response[evidence_position + 9:].strip().split('\n')
                evidences = list(map(lambda s: s[s.find('-') + 2:].strip(), evidences))
                c += 1
                if question_position == -1 or answer_position == -1 or evidence_position == -1:
                    num_of_trials += 1
                    if num_of_trials > 5:
                        break
                    continue
                else:
                    question = response[question_position+9:answer_position].strip()
                    answer = response[answer_position+7:evidence_position].strip()
                    evidence = response[evidence_position+9:].strip()
                    generated.append({"Q":question, "A":answer, "S":evidence})
                    break
        return generated

    def enable_qa(self):
        self.enable_syn_qa = True
        
    def disable_qa(self):
        self.enable_syn_qa = False
