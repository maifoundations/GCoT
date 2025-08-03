import json
import time
import os
import re
import argparse
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def get_box(file_name, num_rounds):
    box_file = json.load(open(f'{file_name}/checked_round{num_rounds}.json', 'r'))
    box_map = {}
    for line in box_file:
        if line.get('sub_ans') == None:
            continue
        sub_target = line['sub_ans']
        box = line['conversations'][1]['value']
        box_map[sub_target] = f'{sub_target} {box}'

    return box_map

def get_box_cot(cot, box_map):
    word_list = cot.split(' ')
    replaced_words = set()  # Set to track words that have been replaced

    for i, word in enumerate(word_list):
        if 'Answer:' in word:
            break

        # Handle words with punctuation separately
        if word.endswith('.'):
            word_root = word[:-1]
            punctuation = '.'
        elif word.endswith(','):
            word_root = word[:-1]
            punctuation = ','
        else:
            word_root = word
            punctuation = ''

        # Replace word only if it hasn't been replaced before
        if word_root in box_map.keys() and word_root not in replaced_words:
            word_list[i] = box_map[word_root] + punctuation
            replaced_words.add(word_root)  # Mark this word as replaced

    return ' '.join(word_list)

def get_cot(file_name, train_file, box_map):
    train_data = json.load(open(f'{file_name}/{train_file}_sampled.json', 'r'))
    gcot_list = []
    for line in train_data:
        id = line['id']
        image = line['image']
        ques = line['conversations'][0]['value']
        answer = line['conversations'][1]['value']
        cot = line['cot']
        gcot = get_box_cot(cot, box_map)

        gcot_list.append(
            {
                "id": id,
                "image": image,
                 "conversations": [
                    {
                        "from": "human",
                        "value": ques
                    },
                    {
                        "from": "gpt",
                        "value": gcot
                    }]
            }
        )
    
    with open(f'{file_name}_gcot.json', 'w') as f:
        json.dump(gcot, f)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--file_name", type=str, default=None)
    parse.add_argument("--num_rounds", type=str, default=None)
    parse.add_argument("--train_file", type=str, default=None)
    args=parse.parse_args()

    file_name = args.file_name
    train_file = args.train_file
    num_rounds = args.num_rounds

    box_map = get_box(file_name, num_rounds)
    get_cot(file_name, train_file, box_map)

