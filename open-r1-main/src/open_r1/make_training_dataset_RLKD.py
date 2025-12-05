import json
from tqdm import tqdm
import argparse

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

    parser = argparse.ArgumentParser(description='Process dataset files')
    parser.add_argument('--input_path', required=True, help='Input JSONL file path')
    parser.add_argument('--train_output', required=True, help='Output train JSONL file path')
    parser.add_argument('--test_output', required=True, help='Output test JSONL file path')
    parser.add_argument('--split_index', type=int, default=90000, help='Split index for train/test')
    args = parser.parse_args()


    data_list = read_jsonl(args.input_path)
    

    train_list = data_list[:args.split_index]
    test_list = data_list[args.split_index:]


    train_count = 0
    with open(args.train_output, 'w') as w:
        for data in tqdm(train_list, desc='Processing train data'):
            for i in range(len(data['generations'])):
                if data['correctness_math_verify'][i] and data['solution'] is not None:
                    data['generation'] = data['generations'][i]
                    json.dump(data, w)
                    w.write('\n')
                    train_count += 1
                    break


    test_count = 0
    with open(args.test_output, 'w') as w:
        for data in tqdm(test_list, desc='Processing test data'):
            for i in range(len(data['generations'])):
                if data['correctness_math_verify'][i] and data['solution'] is not None:
                    data['generation'] = data['generations'][i]
                    json.dump(data, w)
                    w.write('\n')
                    test_count += 1
                    break

    print(f'Train samples saved: {train_count}')
    print(f'Test samples saved: {test_count}')

if __name__ == "__main__":
    main()
