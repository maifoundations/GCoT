import json
import time
import os
import base64
import random
import re
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def is_numeric_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_valid_token(token):
    # A regex pattern to match alphanumeric tokens and tokens with specific symbols ($, ., %, etc.)
    pattern = re.compile(r'^[\w$.\%]+$')
    return pattern.match(token) is not None

def split(file_name, train_file, sample_num):
    train_data = json.load(open(train_file, 'r'))
    random.shuffle(train_data)
    sampled_file = train_data[:sample_num]
    sub_ques_file = f'{file_name}/{train_file}_splited.json'
    if not os.path.exists(sub_ques_file):
        sub_ques = []
        for line in sampled_file:
            idx = line['id']
            image = line['image']
            ques = line['conversations'][0]['value']
            cot = line['cot']

            tokens = word_tokenize(cot)
            tagged_tokens = pos_tag(tokens)

            # Extract nouns and values, with an additional regex filter
            nouns = [word for word, pos in tagged_tokens if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and is_valid_token(word)]
            values = [word for word, pos in tagged_tokens if pos == 'CD' and is_valid_token(word)]

            sub_list = list(set(nouns + values))

            for sub in sub_list:
                if is_numeric_string(sub):
                    sub_prefix = 'Where is the number'
                else:
                    sub_prefix = 'Where is the text'
                sub_ques.append({
                    'idx': idx,
                    'image': image,
                    'question': f'{sub_prefix} {sub}?',
                    'sub': sub
                })

        with open(sub_ques_file, 'w') as f:
            json.dump(sub_ques, f)
        with open(f'{file_name}/{train_file}_sampled_{sample_num}.json', 'w') as f:
            json.dump(sampled_file, f)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--file_name", type=str, default=None)
    parse.add_argument("--train_file", type=str, default=None)
    parse.add_argument("--image", type=str, default=None)
    parse.add_argument("--sample_num", type=int, default=None)
    args=parse.parse_args()

    file_name = args.file_name
    train_file = args.train_file
    sample_num = args.sample_num
    image_path = args.image

    split(file_name, train_file, sample_num, image_path)