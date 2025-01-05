import os
import openai
import tqdm
import json
import argparse


client = openai.Client(api_key=os.getenv('GPT4_API_KEY'))

def get_gpt4_score(question: str, reference: str, pred: str):
    sys_prompt = "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' ."
    prompt = [{"role": "system", "content": sys_prompt,},
    {
        "role": "user",
        "content": "Question: "
        + question
        + "\n"
        + "groundtruth = "
        + reference
        + "\n"
        + "predict_answer = "
        + pred,
    }]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        max_tokens=10,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response = response.choices[0].message.content
    if not response:
        print("Error: No response from GPT-4")
    if 'True' in response or 'true' in response or 'TRUE' in response:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON/JSONL file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the output JSON file.')
    args = parser.parse_args()
    
    file_name, file_ext = os.path.splitext(args.input)
    if file_ext == '.json':
        with open(args.input, 'r') as f:
            samples = json.load(f)
    elif file_ext == '.jsonl':
        samples = []
        with open(args.input, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
    
    # overwrite the output file
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    # resume the output file
    gpt4_scores = []
    num_resumed = 0
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for sample in f.readlines():
                num_resumed += 1
                sample = json.loads(sample)
                if 'qa_pairs' in sample:
                    for qa_pair in sample['qa_pairs']:
                        gpt4_scores.append(qa_pair['scores']['gpt4_score'])
    
    for sample in tqdm.tqdm(samples, total=len(samples), desc='Calculating GPT-4 score'):
        if 'qa_pairs' in sample:
            if num_resumed > 0:
                num_resumed -= 1
                continue
            for qa_pair in sample['qa_pairs']:
                question = qa_pair["Q"]
                reference = qa_pair["A"]
                pred = qa_pair["pred"]
                gpt4_score = get_gpt4_score(question, reference, pred)
                if 'scores' not in qa_pair.keys():
                    qa_pair['scores'] = {}
                    qa_pair['scores']['gpt4_score'] = gpt4_score
                else:
                    qa_pair['scores']['gpt4_score'] = gpt4_score
                gpt4_scores.append(gpt4_score)
            with open(args.output, 'a') as f:
                f.write(json.dumps(sample) + '\n')
    if len(gpt4_scores) > 0 and num_resumed == 0:
        gpt4_score = sum(gpt4_scores) / len(gpt4_scores)
        with open(args.output, 'a') as f:
            f.write(json.dumps({'gpt4_score': gpt4_score}) + '\n')
