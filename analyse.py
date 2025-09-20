import os 
import json
import sys
from glob import glob 
import numpy as np 
import argparse
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
# from utils.misc import extract_characters_regex
import re
from tabulate import tabulate

args = argparse.ArgumentParser()
args.add_argument('--result_dir', type=str, required=True)
args.add_argument('--include', type=str, default="", required=False)
args.add_argument('--save_report', default=False, action='store_true',  required=False)
args.add_argument('--ppl', default=False, action='store_true',  required=False)
args.add_argument('--ppl_ref_dir', type=str, default=None, required=False)

args = args.parse_args()

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]



def ppl(jsonl):
    ppl = list()
    for _, item in jsonl.items():
        ppl.append(item['ppl']) 
    return np.mean(ppl), np.std(ppl), np.max(ppl), np.min(ppl)


def filter_jsonl_by_acc(jsonl, ref_jsonl, filter_key, filter_value):
    filtered_jsonl = dict()
    for key, item in ref_jsonl.items():
        if item[filter_key] == filter_value:
            # keys = list(jsonl.keys())[0]
            # print(keys)
            # print(key)
            # raise
            filtered_jsonl[key] = jsonl[key]
    return filtered_jsonl
   



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
        print('Now analyzing', base_model)
        if args.ppl:
            if args.ppl_ref_dir is not None:
                ref_jsonl = group_jsonls(os.path.join(args.ppl_ref_dir, base_model))
            else:
                ref_jsonl = None

            
            base_ppl_jsonl = group_jsonls(os.path.join(args.result_dir, base_model))
            try:
                base_ppl_jsonl = filter_jsonl_by_acc(base_ppl_jsonl, ref_jsonl, 'acc', 0.0)
            except:
                print(f'{base_model} has no acc == 1.0')
                continue
            ppl_mean, ppl_std, ppl_max, ppl_min = ppl(base_ppl_jsonl)
            print(f'{base_model}: {os.path.join(args.ppl_ref_dir, base_model)} PPL: {ppl_mean:.4f} ± {ppl_std:.4f} (max: {ppl_max:.4f}, min: {ppl_min:.4f})')
            for exp_setting in exp_settings:
                exp_ppl_jsonl = group_jsonls(os.path.join(args.result_dir, exp_setting))
                try:
                    exp_ppl_jsonl = filter_jsonl_by_acc(exp_ppl_jsonl, ref_jsonl, 'acc', 1.0)
                except:
                    print(f'{exp_setting} has no acc == 1.0')
                    continue
                ppl_mean, ppl_std, ppl_max, ppl_min = ppl(exp_ppl_jsonl)
                print(f'{exp_setting}: {os.path.join(args.result_dir, exp_setting)} PPL: {ppl_mean:.4f} ± {ppl_std:.4f} (max: {ppl_max:.4f}, min: {ppl_min:.4f})')
        else:
            for exp_setting in exp_settings:
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

            print('#'*200)
            print('#'*200)





                # print('#'*100)
                
                
