import json
import argparse
from tqdm import tqdm
from Prompt import prompt_math, prompt_code, prompt_science, prompt_puzzle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset with external file paths')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input JSONL file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save processed JSON file')
    return parser.parse_args()

def read_jsonl(file_path):
    results_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                results_list.append(json.loads(line))
            except:
                continue
    return results_list

def main():
    args = parse_arguments()
    
    data_list = read_jsonl(args.input_path)
    
    with open(args.output_path, 'w') as w:
        step = 0
        for data in tqdm(data_list):
            if data['domain'] == 'math':
                prompt = prompt_math
            elif data['domain'] == 'code':
                prompt = prompt_code
            elif data['domain'] == 'puzzle':
                prompt = prompt_puzzle
            else:
                prompt = prompt_science

            response = '[Response]:\n' + data['generated_content'] + '</answer>\n<|endoftext|>'

            data_temp = {
                'system': """You are a helpful AI Assistant. Decompose the problem into multiple steps according to the thinking process for reasoning. Format the thinking content in the form of [sub-think]-[Query]-[Answer] for each step. When the [Answer] of a certain step gets the answer to the original question, stop reasoning, and the last answer should include a complete answer to the original question.
[sub-think] represents the thinking process. It is necessary to analyze the previous and next steps as detailed as possible, explain in detail why this step is done to lead to the following query, and do not mention the answer of this step.
[Query] represents the problem that needs to be solved in the current step, and analyze it as detailed as possible. Each query contains only one question.
[Answer] represents the answer to the Query. Give the answer to the [Query] directly and briefly without any reasoning steps and redundant statements.
You should split it into more [sub-think]-[Query]-[Answer] with finer granularity, but do not repeat the conditions given in the question.""",
                'conversations': [{
                    'from': 'user',
                    'value': f"{prompt}{data['problem']}\nYou should decompose this reasoning:\n{data['deepseek_reasoning']}"
                }, {
                    'from': 'assistant',
                    'value': response
                }],
                'messages': [{
                    'content': f"{prompt}{data['problem']}\nYou should decompose this reasoning:\n{data['deepseek_reasoning']}",
                    'role': 'user'
                }, {
                    'content': response,
                    'role': 'assistant'
                }]
            }

            json.dump(data_temp, w)
            w.write('\n')
            w.flush()  
            
            step += 1
            if step == 1:
                print(data_temp)

if __name__ == "__main__":
    main()
