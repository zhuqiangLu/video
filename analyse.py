import os 
import json
import sys
from glob import glob 
import numpy as np 
import argparse
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
from utils.misc import extract_characters_regex
from tabulate import tabulate

args = argparse.ArgumentParser()
args.add_argument('--result_dir', type=str, required=True)
args.add_argument('--include', type=str, default="", required=False)
args.add_argument('--save_report', default=False, action='store_true',  required=False)
args = args.parse_args()


def acc(jsonl):
    acc = list()
    for _, item in jsonl.items():
        acc.append(item['acc']) 
    return np.mean(acc)

def group_jsonls(jsonl_dir):
    all_jsonl = glob(os.path.join(jsonl_dir, '*.jsonl'))
    tmp_dict = dict()
    for jsonl in all_jsonl:
        with open(jsonl, 'r') as f:
            for line in f:
                item = json.loads(line)
                key = item['prompt']+item['video_path']
                tmp_dict[key] = item 

    return tmp_dict

def parse_config(setting_path):
    with open(setting_path, 'r') as f:
        config = json.load(f)
    return config 
                
def consistency(jsonl1, jsonl2):
    # make sure jsonl1 is the base exp
    consistency_report = list()
    consistency_pair = list()
    hit_rate = 0
    for key, item in jsonl1.items():
        if key in jsonl2:
            hit_rate += 1
            consistency_pair.append((item['pred'], jsonl2[key]['pred']))
            consistency_report.append(
                {
                    'videopath': item['video_path'],
                    'pred_base': item['pred'],
                    'pred_exp': jsonl2[key]['pred'],
                    'prompt': item['prompt'],
                    'gt': item['gt'],
                }
            )

    cs = 0.
    for cp in consistency_pair:
        pred1, pred2 = cp[0], cp[1]
        pred1 = extract_characters_regex(pred1)
        pred2 = extract_characters_regex(pred2)
        # print(cp[0], cp[1])
        # print(pred1, pred2)
        # raise
        if pred1 == pred2:
            cs += 1
        
    cs_score = cs / len(consistency_pair) if len(consistency_pair) > 0 else 0.
    hit_rate /= len(jsonl1)
    return cs_score, hit_rate, consistency_report

def build_table(settings):
    settings = settings.split('|')[1:] 
    table = defaultdict(list)
    for s in settings:
        if ":" not in s:
            table["frame_setting"].append(s) 
        else:
            key, value = s.split(":")
            table[key].append(value)

    header = list()
    data = list()
    for k, v in table.items(): 
        data.append([ "|".join(v)])
        header.append(k)
    return header, data

    


if __name__ == '__main__':
    
    all_jsonl = glob(os.path.join(args.result_dir, '*/*.jsonl')) 
    all_exp = defaultdict(list)
    
    for jsonl in all_jsonl:
        parent_dir = os.path.dirname(jsonl).split('/')[-1]
        if args.include in parent_dir:
            all_exp[parent_dir].append(jsonl)

    # find the base model name 
    all_base_model = defaultdict(list)
    all_exp_name = list(all_exp.keys())
    for exp_name in all_exp_name:
        base_model_name = exp_name.split('|')[0]
        all_base_model[base_model_name].append(exp_name)


    for base_model, exp_settings in all_base_model.items():
        # base_model_jsonl = all_exp[base_model]
        for exp_setting in tqdm(exp_settings):
            if base_model != exp_setting:
                base_json_dir = os.path.join(args.result_dir, base_model)
                setting_json_dir = os.path.join(args.result_dir, exp_setting)
                base_config = parse_config(os.path.join(base_json_dir, 'config.json'))
                
                base_json_dict = group_jsonls(base_json_dir)
                setting_json_dict = group_jsonls(setting_json_dir)
               
                headers, data = build_table(exp_setting, ) 
                cs, hit_rate, consistency_report = consistency(base_json_dict, setting_json_dict)
                headers.append('# items (base|exp)')
                data[0].append([f"{len(base_json_dict)}|{len(setting_json_dict)}"])
                headers.append('# acc (base|exp)')
                data[0].append([f"{acc(base_json_dict):.4f}|{acc(setting_json_dict):.4f}"])
                headers.append('# consistency')
                data[0].append([f"{cs:.4f}"])
                headers.append('# hit rate')
                data[0].append([f"{hit_rate:.4f}"])
                print(tabulate(data, headers=headers, tablefmt="grid"), )
                if args.save_report:
                    print(f'saving report to {os.path.join(args.result_dir, f"{base_model}_vs_{exp_setting}.json")}')
                    with open(os.path.join(args.result_dir, f'{base_model}_vs_{exp_setting}.json'), 'w') as f:
                        json.dump(consistency_report, f, indent=4)





                print('#'*100)
                
                
