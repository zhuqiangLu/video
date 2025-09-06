import json
import re

from PIL import Image
import random
# from GQA_TEMPLATE import GQA_TEMPLATE
from tqdm import tqdm 
import os
from dataloader import data_builder
import torch
import numpy as np

# GQA_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION]. Only output the corresponding letter of the option.
# """


GQA_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION]. Answer the question by outputting the corresponding letter of the option, then briefly explain the reason.
"""


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


def append_to_jsonl(file_path, data):
    
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)  
            f.write(json_line + '\n')  
    except Exception as e:
        print(f"写入文件时发生错误: {e}")





def run_experiment(
    exp_configurator,
    get_inputs_func,
    inference_func,
    dataset,
    model_base,
    setup_model_func,
    device,
    resume,
    max_new_tokens,
    debug=False,
    **kwargs
):
    model, processor = setup_model_func(model_base, device)

    log_path = exp_configurator.get_log_path() 

    # rm log_path 
    resume_idx = 0
    if resume:
        assert os.path.exists(log_path), "log_path does not exist" 
        with open(log_path, 'r') as f:
            for line in f:
                resume_idx += 1
        print(f"resume from {resume_idx}")
    else:
        if os.path.exists(log_path):
            os.remove(log_path)

    
    pbar = tqdm(dataset)
    for idx, item in enumerate(pbar):
        if idx < resume_idx:
            continue

        question, options, answer, frames, _ = exp_configurator.configure_inputs(item)
        
        if debug:
            os.makedirs(f"{exp_configurator._log_root}/debug/{kwargs.get('cur_gpu', 0)}-{idx}", exist_ok=True)
            with open(f"{exp_configurator._log_root}/debug/{kwargs.get('cur_gpu', 0)}-{idx}/item.json", "w") as f:
                extra = {
                    "extra_video_path": item['extra_video_path'],
                }
                json.dump(extra, f, indent=4)
            frames_dir = f"{exp_configurator._log_root}/debug/{kwargs.get('cur_gpu', 0)}-{idx}/frames"
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                frame.save(os.path.join(frames_dir, f"frame_{i}.jpg"))


        video_path = item['video_path']
        example_prompt = GQA_TEMPLATE.replace("[QUESTION]", question)
        prompt = example_prompt.replace("[OPTION]", str(options))

        accs = []
  
        inputs = get_inputs_func(prompt, frames, processor)
  
        try:

            if inference_func is None:
                inputs = inputs.to(device)
                input_ids = inputs.input_ids
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, )
                generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
                pred = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

            else:
                pred = inference_func(model, processor, inputs, max_new_tokens=max_new_tokens, use_cache=True)

           
            acc = 0.0
            if extract_characters_regex(answer) == extract_characters_regex(pred):
                acc = 1.0

            if answer in pred[:1]:  
                acc = 1.0

            accs.append(acc)

            
            
            item_res = {'video_path': video_path, 'prompt':prompt, 'gt':answer, 'pred':pred, 'acc':acc }
            append_to_jsonl(log_path, item_res)
            
            pbar.set_postfix({'accuracy': sum(accs)/len(accs), "gpu_id": kwargs.get('cur_gpu', 0)})

        except Exception as e: 
            print(f"Error: {e}")
            print(f"video_path: {video_path}")


            



    

    