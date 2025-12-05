import json
from tqdm import tqdm
import argparse
from Prompt import prompt_math, prompt_code, prompt_science, prompt_puzzle
from openai import OpenAI
import joblib
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset paths')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the original dataset JSON file')
    parser.add_argument('--random_sampled_path', type=str, required=True,
                        help='Path to save/load the randomly sampled joblib file')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the processed results')
    return parser.parse_args()


def call_qwen(instruction, client):
    success_flag = 0
    while success_flag == 0:
        try:
            chat_response = client.chat.completions.create(model="gpt-4o",
                                                           messages=[{"role": "user",
                                                                      "content": instruction}])
            success_flag = 1
            return chat_response.choices[0].message.content.split(instruction)[-1]
        except:
            print("request fail")
            success_flag = 0


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    args = parse_arguments()

    openai_api_key = "your api key"
    openai_api_base = 'your openai api base'
    client_gpt4 = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    # load
    data_list = read_jsonl(args.dataset_path)
    random.shuffle(data_list)

    # sample and save
    joblib.dump(data_list, args.random_sampled_path)
    data_list = joblib.load(args.random_sampled_path)

    # process and save
    a = set()
    with open(args.save_path, 'a') as w:
        step = 0
        for data in tqdm(data_list):
            a.add(data['domain'])

            if data['domain'] == 'math':
                prompt = prompt_math
            elif data['domain'] == 'code':
                prompt = prompt_code
            elif data['domain'] == 'puzzle':
                prompt = prompt_puzzle
            else:
                prompt = prompt_science

            print('step is {}'.format(step))
            print(data['problem'])
            input = prompt + data['problem'] + '\nYou can reference this reasoning:\n' + data[
                'deepseek_reasoning'] + '\n[Response]: \n'
            response = call_qwen(instruction=input, client=client_gpt4)
            print(response)
            data['generated_content'] = response
            json.dump(data, w)
            w.write('\n')
            w.flush()
            step += 1
