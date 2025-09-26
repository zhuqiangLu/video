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
import matplotlib.pyplot as plt



args = argparse.ArgumentParser()
args.add_argument('--result_dir', type=str, required=True)
args.add_argument('--include', type=str, default="", required=False)
args.add_argument('--save_report', default=False, action='store_true',  required=False)
args.add_argument('--ppl', default=False, action='store_true',  required=False)
args.add_argument('--ppl_ref_dir', type=str, default=None, required=False)
args.add_argument('--filter_key', type=str, default=None, required=False)
args.add_argument('--filter_value', type=float, default=None, required=False)
args.add_argument('--draw_ppl_plot', action='store_true', default=False, required=False)
args = args.parse_args()


def draw_ppl_plot(ppl_data_list, base_model, plot_dir):

    ppl_mean_exp = list()
    ppl_std_exp = list()
    exp_name_exp = list()
    ppl_min_exp = list()
    ppl_max_exp = list()    

    for ppl_data in ppl_data_list:
        [exp_name, ppl_mean_, ppl_std, ppl_max, ppl_mn] = ppl_data
        exp_name_exp.append(exp_name)
        ppl_mean_exp.append(ppl_mean_)
        ppl_std_exp.append(ppl_std)
        ppl_min_exp.append(ppl_mn)
        ppl_max_exp.append(ppl_max)

    ppl_mean_exp = np.array(ppl_mean_exp)
    lower_err = ppl_mean_exp - ppl_min_exp
    upper_err = ppl_max_exp - ppl_mean_exp
    yerr = [lower_err, upper_err]

    fig, ax = plt.subplots(figsize=(18, 5))


    bars = ax.bar(exp_name_exp, ppl_mean_exp, yerr=yerr, capsize=6, edgecolor='black', alpha=0.9)
    # add mean markers
    for bar, mean in zip(bars, ppl_mean_exp):
        ax.plot(bar.get_x() + bar.get_width()/2, mean, 'o', markersize=5)

    ax.set_yscale("log")
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title(f"PPL per Setting with Min–Max Range — {base_model}")
    ax.grid(True, which="both", axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    # plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"single_ppl.png"))
    plt.close()

def draw_relative_ppl_plot(ppl_data_list, base_model, plot_dir):

    # make list a dict 
    exp_data_dict = dict()
    for ppl_data in ppl_data_list:
        [exp_name, ppl_mean_exp, ppl_std_exp, ppl_max_exp, ppl_min_exp] = ppl_data
        exp_data_dict[exp_name] = {
            'mean': ppl_mean_exp,
            'std': ppl_std_exp,
            'max': ppl_max_exp,
            'min': ppl_min_exp
        }

    # Find the base model data
    base_ppl = None
    for exp_name, vals in exp_data_dict.items():
        if exp_name == base_model:
            base_ppl = vals["mean"]
            break
    
    if base_ppl is None:
        print(f"Warning: Base model {base_model} not found in data")
        return
    
    rel_means = {}
    rel_stds = {}
    for exp_name, vals in exp_data_dict.items():
        if exp_name == base_model:
            continue
        # Calculate relative change from base

        rel_change = (vals["mean"] - base_ppl) / base_ppl * 100.0
        print(exp_name, vals["mean"], base_ppl, rel_change)
        
        rel_means[exp_name] = rel_change
        # std propagated as % of base for visualizing uncertainty scale
        rel_stds[exp_name] = vals["std"] / base_ppl * 100.0

    # Create the plot
    exp_names = list(rel_means.keys())
    x = np.arange(len(exp_names))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each experiment
    bars = ax.bar(x, [rel_means[name] for name in exp_names], 
                  yerr=[rel_stds[name] for name in exp_names], 
                  capsize=5, edgecolor='black', alpha=0.7)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel("Relative Δ PPL(exp-base/base) vs Base")
    ax.set_title(f"Effect of Perturbations on PPL (Relative to {base_model})")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"relative_ppl.png"))
    plt.close()


def acc(jsonl):
    acc = list() 
    for key, val in jsonl.items():
    
        acc.append(val['acc']) 

    return np.mean(acc)

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


def filter_jsonl_by_key(jsonl, ref_jsonl, filter_key, filter_value):
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

    ppl_mean = list()
    ppl_max = list()
    ppl_min = list()
    ppl_std = list()
    settings = list() 
    for base_model, exp_settings in all_base_model.items():
        # base_model_jsonl = all_exp[base_model]
        print('Now analyzing', base_model)
        if args.ppl:
            if args.ppl_ref_dir is not None:
                ref_jsonl = group_jsonls(os.path.join(args.ppl_ref_dir, base_model))
            else:
                ref_jsonl = None

            ppl_data = list()
            for exp_setting in exp_settings:
                exp_ppl_jsonl = group_jsonls(os.path.join(args.result_dir, exp_setting))
                if args.filter_key is not None and args.filter_value is not None:
                    try:
                        
                        exp_ppl_jsonl = filter_jsonl_by_key(exp_ppl_jsonl, ref_jsonl, args.filter_key, args.filter_value)
                    except:
                        print(f'{exp_setting} has no acc == 1.0')
                        continue

             
                ppl_mean_exp, ppl_std_exp, ppl_max_exp, ppl_min_exp = ppl(exp_ppl_jsonl)

                exp_name = exp_setting.split('|')
                if len(exp_name) > 1:
                    exp_name = "\n".join(exp_name[1:])
                else:
                    exp_name = exp_name[0]


                data = [exp_name, ppl_mean_exp, ppl_std_exp, ppl_max_exp, ppl_min_exp]
                ppl_data.append(data)
                

            headers = ['exp_name', 'ppl_mean', 'ppl_std', 'ppl_max', 'ppl_min']
            print(tabulate(ppl_data, headers=headers, tablefmt="grid"), )

            if args.draw_ppl_plot:
                plot_dir = os.path.join('plots', base_model)
                os.makedirs(plot_dir, exist_ok=True)
                
                draw_ppl_plot(ppl_data, base_model, plot_dir)
                draw_relative_ppl_plot(ppl_data, base_model, plot_dir)
                


        else:
            acc_data = list()
            base_json_dir = os.path.join(args.result_dir, base_model)                
            base_json_dict = group_jsonls(base_json_dir)

            acc_data.append([base_model, acc(base_json_dict), len(base_json_dict), '-', '-'])
            
            
            for exp_setting in exp_settings:
                if base_model == exp_setting:
                    continue 
                # if base_model != exp_setting:
                exp_json_dir = os.path.join(args.result_dir, exp_setting)                
                exp_json_dict = group_jsonls(exp_json_dir)
            
                # headers, data = build_table(exp_setting, ) 
                exp_name = exp_setting.split('|')
                if len(exp_name) > 1:
                    exp_name = "\n".join(exp_name[1:])
                else:
                    exp_name = exp_name[0]


                
                cs, hit_rate, _ = consistency(base_json_dict, exp_json_dict)

                data = [exp_name, acc(exp_json_dict), len(exp_json_dict ), cs, hit_rate]
                acc_data.append(data)
            headers = ['exp_name', 'acc', '# items', 'cs', 'hit_rate']
            print(tabulate(acc_data, headers=headers, tablefmt="grid"), )   
               




                # print('#'*100)
                
                
