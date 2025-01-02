import os
import re
import json
import numpy as np
import argparse


def concordinary_index(pred_list, right_list):
    pred_set = set()
    right_set = set()
    error = False
    for i in range(len(pred_list) - 1):
        if min(pred_list) == 1:
            for j in range(i + 1, len(pred_list)):
                pred_set.add((pred_list[i] - 1, pred_list[j] - 1))
        else:
            for j in range(i + 1, len(pred_list)):
                pred_set.add((pred_list[i], pred_list[j]))
    if len(pred_list) != len(right_list):
        error = True
    for i in range(len(right_list) - 1):
        for j in range(i + 1, len(right_list)):
            right_set.add((right_list[i], right_list[j]))
    return len(right_set & pred_set) / len(right_set), error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred')
    parser.add_argument('--strict_mode', action='store_true')
    args = parser.parse_args()
    with open(args.pred, 'r') as f:
        ext = os.path.splitext(args.pred)[1]
        if ext == '.json':
            data = json.load(f)
        elif ext == '.jsonl':
            data = list(map(json.loads, f.readlines()))
        else:
            raise ValueError(f"Unknown file extension {ext}")
    scores = []
    errors = []
    scores_wo_error = []
    for sample in data:
        if 'qa_pairs' in sample:
            for qa in sample['qa_pairs']:
                answer_pattern = r"\[[0-9]+\](?: < \[[0-9]+\])*"
                answer = qa['answers']
                if 0 in answer:
                    answer = list(map(lambda x: x + 1, qa['answers']))
                if args.strict_mode:
                    pred = re.findall(answer_pattern, qa['pred'])
                    if len(pred) == 0:
                        errors.append(1)
                        scores.append(0)
                    else:
                        pred = list(map(int, re.findall(r"[0-9]+", pred[0])))
                        score, error = concordinary_index(pred, answer)
                        if error:
                            errors.append(1)
                            scores.append(0)
                        else:
                            errors.append(0)
                            scores.append(score)
                            scores_wo_error.append(score)
                else:
                    pred = list(map(int, re.findall(r"[0-9]+", qa['pred'])))
                    score, error = concordinary_index(pred, answer)
                    if error:
                        errors.append(1)
                        scores.append(0)
                    else:
                        errors.append(0)
                        scores.append(score)
                        scores_wo_error.append(score)
        elif 'pred' in sample and ('answers' in sample or 'answer' in sample):
            if 'answer' in sample:
                answer = sample['answer']
            else:
                answer = sample['answers']
            if 0 in answer:
                answer = [a + 1 for a in answer]
            if args.strict_mode:
                answer_pattern = r"\[[0-9]+\](?: [<>] \[[0-9]+\])*"
                if 0 in answer:
                    answer = list(map(lambda x: x + 1, answer))
                pred = re.findall(answer_pattern, sample['pred'])
                if len(pred) == 0:
                    errors.append(1)
                    scores.append(0)
                else:
                    pred = list(map(int, re.findall(r"[0-9]+", pred[0])))
                    score, error = concordinary_index(pred, answer)
                    if error:
                        errors.append(1)
                        scores.append(0)
                    else:
                        errors.append(0)
                        scores.append(score)
                        scores_wo_error.append(score)
            else:
                pred = list(map(int, re.findall(r"[0-9]+", sample['pred'])))
                score, error = concordinary_index(pred, answer)
                if error:
                    errors.append(1)
                    scores.append(0)
                else:
                    errors.append(0)
                    scores.append(score)
                    scores_wo_error.append(score)
    print("Average score:", np.mean(scores))
    print("Error rate:", np.mean(errors))
    print("Average score w/o error:", np.mean(scores_wo_error))


if __name__ == '__main__':
    main()
