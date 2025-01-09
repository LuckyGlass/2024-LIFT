import argparse
import json
import tqdm
from transformers import AutoTokenizer


LOOGLEFORMAT_COT = """The article {title}:
{input}
Please recall one or several original sentences from article {title} as evidence, and then answer the question solely based on this evidence.
Please answer in the following format:

# Evidence:
- [evidence 1]
- [evidence 2]
...
# Answer:
[answer]
# End of answer

Please DON'T output quotes when outputing evidences.
# Question:
{question}
# Evidence:
"""

parser = argparse.ArgumentParser()
parser.add_argument('-S', '--source', help="The parsed output file.")
parser.add_argument('-D', '--dest', help="The target file.")
parser.add_argument('--model_max_length', type=int)
parser.add_argument('--tokenizer_name_or_path')
args = parser.parse_args()
source_file = args.source
dest_file = args.dest
model_max_length = args.model_max_length
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
mixin = tokenizer("...", add_special_tokens=False)['input_ids']

with open(source_file, 'r') as f:
    data = json.load(f)

for sample in tqdm.tqdm(data):
    for qa_pair in sample['qa_pairs']:
        if 'S_pred' not in qa_pair:
            continue
        input_text = LOOGLEFORMAT_COT.format(title=sample['title'], input=sample['input'], question=qa_pair['Q'])
        input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
        icl_text = tokenizer.decode(input_ids)
        qa_pair['S_is_in_ICLcontext'] = qa_pair['S'] in icl_text
        for i in range(len(qa_pair['S_pred'])):
            qa_pair['S_pred'][i] = {
                'is_in_fullcontext': qa_pair['S_pred'][i] in sample['input'],
                'is_in_ICLcontext': qa_pair['S_pred'][i] in icl_text,
                'text': qa_pair['S_pred'][i]
            }

with open(dest_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
