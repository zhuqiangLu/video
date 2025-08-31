import os 
import json
import sys
from glob import glob 
import numpy as np 
import argparse
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
args = argparse.ArgumentParser()
args.add_argument('--result_dir', type=str, required=True)
args.add_argument('--include', type=str, default="", required=False)
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
                
def consistency(jsonl1, jsonl2):
    consistency_pair = list()
    hit_rate = 0
    for key, item in jsonl1.items():
        if key in jsonl2:
            hit_rate += 1
            consistency_pair.append((item['pred'], jsonl2[key]['pred']))

    cs = 0.
    for cp in consistency_pair:
        if cp[0] == cp[1]:
            cs += 1
        
    cs_score = cs / len(consistency_pair) if len(consistency_pair) > 0 else 0.
    hit_rate /= len(jsonl1)
    return cs_score, hit_rate



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
        base_model_name = exp_name.split('_')[0]
        all_base_model[base_model_name].append(exp_name)


    for base_model, exp_settings in all_base_model.items():
        # base_model_jsonl = all_exp[base_model]
        for exp_setting in tqdm(exp_settings):
            if base_model != exp_setting:
                base_json_dir = os.path.join(args.result_dir, base_model)
                setting_json_dir = os.path.join(args.result_dir, exp_setting)
                base_json_dict = group_jsonls(base_json_dir)
                setting_json_dict = group_jsonls(setting_json_dir)

                print(f'{base_model} has {len(base_json_dict)} items and {exp_setting} has {len(setting_json_dict)} items')
                cs, hit_rate = consistency(base_json_dict, setting_json_dict)
                print(f'{base_model} {acc(base_json_dict)} -> {exp_setting} {acc(setting_json_dict)}: {cs} {hit_rate}')
                print('-'*100)
                
                

    # find the max num frames
    
    # acc = list()
    # for jonsl in all_jonsl:
    #     with open(jonsl, 'r') as f:
    #         for line in f:
    #             item = json.loads(line)
    #             acc.append(item['acc']) 
    # print(np.mean(acc))    
