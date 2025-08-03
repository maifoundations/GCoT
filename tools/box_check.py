import os
import json
import argparse

def get_acc(box_check_file, base_path):

    box_check_result = open(box_check_file, 'r').readlines()
    correct_line = []
    count = 0
    correct = 0
    for i, line in enumerate(box_check_result):
        line = json.loads(line)
        self_out = line['text']
        if ':' in self_out:
            self_out = self_out.split(':')[1]
        self_out = self_out.strip()
        answer = line['sub']
        # focus = line['focus']

        if self_out.lower().strip() == answer.lower().strip():
            correct+=1
            correct_line.append(line)
        count+=1

    acc_rate = correct/count
    print(acc_rate)
    with open(os.path.join(base_path, 'box_acc.txt'), 'a') as f:
        f.write('*************************')
        f.write('\n')
        f.write('correct_num:')
        f.write(str(correct))
        f.write('\n')
        f.write('correct_rate:')
        f.write(str(acc_rate))
        f.write('\n')
        f.write('*************************')

    return correct_line

def get_correct(correct_list, save_path, sample_file):
    train_box_table = []
    id_list = []
    for line in correct_list:
        question_id = line['question_id']
        image  = line['image']
        question = line['prompt']
        answer = line['box']
        sub_answer = line['sub']
        # ques_type = line['ques_type']

        train_box_table.append({
                    "id": question_id,
                    "image": image, 
                    "conversations": [
                        {'from': 'human', 'value': f"<image>\n{question}"},
                        {'from': 'gpt', 'value': f"{answer}"},
                    ],
                    "sub_ans": sub_answer,
                })
        id_list.append(question_id)

    round_num = save_path.split('round')[-1].split('.')[0]
    if round_num == '0':
        if sample_file is not None:
            simple_qa = json.load(open(sample_file, 'r'))
            if 'tab' in sample_file or 'chart' in sample_file:
                for line in simple_qa:
                    line['conversations'][0]['value'] = line['conversations'][0]['value'] + '\nAnswer the question using a single word or phrase.'
                train_box_table += simple_qa
            elif 'dia' in sample_file:
                for line in simple_qa:
                    ques, option = line['conversations'][0]['value'].split('Options:')
                    ans_set = option.split('\n')
                    ans_map={}
                    for ans in ans_set:
                        if 'A. ' in ans:
                            ans_map['A'] = ans.split('A. ')[-1]
                        if 'B. ' in ans:
                            ans_map['B'] = ans.split('B. ')[-1]
                        if 'C. ' in ans:
                            ans_map['C'] = ans.split('C. ')[-1]
                        if 'D. ' in ans:
                            ans_map['D'] = ans.split('D. ')[-1]
                    line['conversations'][0]['value'] = line['conversations'][0]['value'] + '\nAnswer the question using a single word or phrase.'
                    line['conversations'][1]['value'] = ans_map[line['conversations'][1]['value']]
                train_box_table += simple_qa
            else:
                for line in simple_qa:
                    line['conversations'][0]['value'] = line['conversations'][0]['value'] + '\nAnswer the question using a single word or phrase.'
                train_box_table += simple_qa
    else:
        former_file = json.load(open(save_path.replace(f'round{round_num}', f'round{str(int(round_num) - 1)}'), 'r'))
        for line in former_file:
            id = line['id']
            if not id in id_list:
                train_box_table.append(line)
        
    with open(save_path, 'w') as f:
        json.dump(train_box_table, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_file", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--base_path", type=str)
    parser.add_argument('--sampled_file', type=str, default=None)
    parser.add_argument('--final', type=bool, default=False)

    args = parser.parse_args()
    box_check_file = args.check_file
    save_path = args.save_path
    base_path = args.base_path
    sampled_file = args.sampled_file
    final = args.final

    correct_list = get_acc(box_check_file, base_path)
    get_correct(correct_list, save_path, sampled_file)