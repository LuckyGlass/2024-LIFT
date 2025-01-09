"""
Metrics:
1. in-ICL x Correct     : in-ICL
2. out-ICL x Correct    : out-ICL
3. in-ICL               : All
4. Correct              : All
5. in-ICL x Correct     : Correct
6. in-ICL x Retrieved   : in-ICL
7. out-ICL x Retrieved  : out-ICL
8. Retrieved            : All
9. Retrieved x Correct  : Retrieved
10: Fail                : All
"""
import argparse
import json
import os
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

import re
from Levenshtein import distance

def preprocess_sentence(sentence):
    # 转为小写并去掉标点符号
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # 去掉标点符号
    sentence = re.sub(r'\s+', ' ', sentence).strip()  # 去掉多余空格
    return sentence

def are_sentences_similar(sentence1, sentence2, threshold=0.5):
    s1 = preprocess_sentence(sentence1)
    s2 = preprocess_sentence(sentence2)
    edit_distance = distance(s1, s2)
    max_len = max(len(s1), len(s2))
    normalized_distance = edit_distance / max_len if max_len > 0 else 0
    return normalized_distance <= threshold


parser = argparse.ArgumentParser()
parser.add_argument('-S', '--source', help="The parsed output file.")
parser.add_argument('-D', '--dest', help="The target file.")
parser.add_argument('-G', '--gpt4score', help="The gpt4score file.")
parser.add_argument('--model_max_length', type=int)
parser.add_argument('--tokenizer_name_or_path')
parser.add_argument('--example_dir')
args = parser.parse_args()
gpt4score_file = args.gpt4score
source_file = args.source
dest_file = args.dest
example_dir = args.example_dir
model_max_length = args.model_max_length
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
mixin = tokenizer("...", add_special_tokens=False)['input_ids']

try:
    with open(source_file, 'r') as f:
        data = json.load(f)
except json.decoder.JSONDecodeError:
    with open(source_file, 'r') as f:
        data = [json.loads(s) for s in f]
if gpt4score_file is None:
    data_gpt4score = [None for _ in range(len(data))]
else:
    with open(gpt4score_file, 'r') as f:
        data_gpt4score = [json.loads(s) for s in f]

num_ic = 0
num_i = 0
num_oc, example_oc = 0, []
num_o = 0
num_a = 0
num_c = 0
num_ir = 0
num_or, example_or = 0, []
example_R = []
num_r = 0
num_rc = 0
example_rC = []
num_f, example_f = 0, []
for sample, sample_gpt4score in tqdm.tqdm(zip(data, data_gpt4score)):
    if sample_gpt4score is None:
        qa_pairs_gpt4score = [None] * len(sample['qa_pairs'])
    else:
        qa_pairs_gpt4score = [q['scores']['gpt4_score'] for q in sample_gpt4score['qa_pairs']]
    for qa_pair, gpt4score in zip(sample['qa_pairs'], qa_pairs_gpt4score):
        input_text = LOOGLEFORMAT_COT.format(title=sample['title'], input=sample['input'], question=qa_pair['Q'])
        input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
        icl_text = tokenizer.decode(input_ids)
        # attributes
        correct = gpt4score
        in_context = qa_pair['S'] in icl_text
        out_context = not in_context
        if 'S_pred' in qa_pair:
            retrieved = any([are_sentences_similar(s, qa_pair['S']) for s in qa_pair['S_pred']])
            fail = False
        else:
            retrieved = False
            fail = True
        #
        qa_pair['S_is_in_ICLcontext'] = in_context
        qa_pair['Retrieved'] = retrieved
        if 'S_pred' in qa_pair:
            for i in range(len(qa_pair['S_pred'])):
                qa_pair['S_pred'][i] = {
                    'is_in_fullcontext': qa_pair['S_pred'][i] in sample['input'],
                    'is_in_ICLcontext': qa_pair['S_pred'][i] in icl_text,
                    'text': qa_pair['S_pred'][i]
                }
        if fail: num_f += 1
        if in_context and correct: num_ic += 1
        if in_context: num_i += 1
        if out_context and correct: num_oc += 1
        if out_context: num_o += 1
        num_a += 1
        if correct: num_c += 1
        if out_context and retrieved: num_or += 1
        if in_context and retrieved: num_ir += 1
        if retrieved: num_r += 1
        if retrieved and correct: num_rc += 1
        # Save examples
        example = {
            'input': sample['input'],
            'Q': qa_pair['Q'],
            'A': qa_pair['A'],
            'A_pred': qa_pair.get('A_pred', None),
            'S': qa_pair['S'],
            'S_pred': qa_pair.get('S_pred', None),
            'pred': qa_pair.get('pred', None)
        }
        if not fail:
            if out_context and correct and len(example_oc) <= 10: example_oc.append(example)
            if out_context and retrieved and len(example_or) <= 10: example_or.append(example)
            if not retrieved and len(example_R) <= 10: example_R.append(example)
            if retrieved and not correct and len(example_rC) <= 10: example_rC.append(example)
        elif len(example_f) <= 10: example_f.append(example)

print(f"Fail                : All       = {num_f:4d} : {num_a:4d} = {num_f / num_a * 100:0.2f}")
print(f"Correct             : All       = {num_c:4d} : {num_a:4d} = {num_c / num_a * 100:0.2f}")
print(f"in-ICL x Correct    : in-ICL    = {num_ic:4d} : {num_i:4d} = {num_ic / num_i * 100:0.2f}")
print(f"out-ICL x Correct   : out-ICL   = {num_oc:4d} : {num_o:4d} = {num_oc / num_o * 100:0.2f}")
print(f"Retrieved x Correct : Retrieved = {num_rc:4d} : {num_r:4d} = {float('nan') if num_r == 0 else num_rc / num_r * 100:0.2f}")
print(f"in-ICL              : All       = {num_i:4d} : {num_a:4d} = {num_i / num_a * 100:0.2f}")
print(f"Retrieved           : All       = {num_r:4d} : {num_a:4d} = {num_r / num_a * 100:0.2f}")
print(f"in-ICL x Retrieved  : in-ICL    = {num_ir:4d} : {num_i:4d} = {num_ir / num_i * 100:0.2f}")
print(f"out-ICL x Retrieved : out-ICL   = {num_or:4d} : {num_o:4d} = {num_or / num_o * 100:0.2f}")
print(f"in-ICL x Correct    : Correct   = {num_ic:4d} : {num_c:4d} = {num_ic / num_c * 100:0.2f}")

if example_dir is not None:
    os.makedirs(example_dir, exist_ok=True)
    with open(os.path.join(example_dir, 'OutContext-Correct.json'), 'w', encoding='utf-8') as f:
        json.dump(example_oc, f, indent=4, ensure_ascii=False)
    with open(os.path.join(example_dir, 'OutContext-Retrieved.json'), 'w', encoding='utf-8') as f:
        json.dump(example_or, f, indent=4, ensure_ascii=False)
    with open(os.path.join(example_dir, 'Retrieved-Wrong.json'), 'w', encoding='utf-8') as f:
        json.dump(example_rC, f, indent=4, ensure_ascii=False)
    with open(os.path.join(example_dir, 'NotRetrieved.json'), 'w', encoding='utf-8') as f:
        json.dump(example_R, f, indent=4, ensure_ascii=False)
    with open(os.path.join(example_dir, 'Fail.json'), 'w', encoding='utf-8') as f:
        json.dump(example_f, f, indent=4, ensure_ascii=False)
with open(dest_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
