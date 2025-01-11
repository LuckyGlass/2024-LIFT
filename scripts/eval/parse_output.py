import argparse
import json
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-S', '--source', help="Original JSONL file.")
parser.add_argument('-D', '--dest', help="Target JSON file.")
args = parser.parse_args()
source_path = args.source
dest_path = args.dest


def parse_pred(pred: str):
    if "# Answer:" not in pred:
        return None
    answer_pos = pred.find("# Answer:")
    end_of_answer_pos = pred.find("# End of answer")
    if end_of_answer_pos == -1:  # may exceed max_new_tokens
        end_of_answer_pos = len(pred)
    answer = pred[answer_pos + len("# Answer:"):end_of_answer_pos].strip()
    evidence_text = pred[:answer_pos].strip()
    evidences = evidence_text.split('\n')
    if any([e[:2] != '- ' for e in evidences]):
        return None
    evidences = [e[2:].strip() for e in evidences]
    return answer, evidences


with open(source_path, 'r') as f:
    data = [json.loads(s) for s in f]
for sample in tqdm.tqdm(data):
    for qa_pair in sample['qa_pairs']:
        pred = qa_pair.pop('pred')
        result = parse_pred(pred)
        if result is None:
            print('-' * 10)
            print(pred)
            print('=' * 10)
            qa_pair['pred'] = pred
        else:
            qa_pair['A_pred'] = result[0]
            qa_pair['S_pred'] = result[1]

with open(dest_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
