import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import re
import pickle
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import random
import os
import json
from math import ceil
# from eval_prompts import GQA_THINK_ANSWER_GLUE as GQA_TEMPLATE
# from eval_prompts import GQA_ANSWER as GQA_TEMPLATE_NO_TAG
import transformers
import matplotlib.pyplot as plt
from decord import VideoReader, cpu
from PIL import Image
from dataloader import build_data

from qwen_vl_utils import process_vision_info

GQA_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION]. Only output the corresponding letter of the option.
"""





def encode_video(target_video_path, extra_video_paths, max_num_frames=10, combine_type='target_first', return_path=False):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if combine_type == 'target_first':
        video_paths = [target_video_path, *extra_video_paths]
    elif combine_type == 'target_last':
        video_paths = [*extra_video_paths, target_video_path]
    elif combine_type == 'random':
        video_paths = [target_video_path, *extra_video_paths]
        random.shuffle(video_paths)
    else:
        raise ValueError(f"Invalid combine_type: {combine_type}")

    if return_path:
        return video_paths
    
    all_frames = []
    for video_path in [target_video_path, *extra_video_paths]:

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > max_num_frames:
            frame_idx = uniform_sample(frame_idx, max_num_frames)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        all_frames.extend(frames) 

    all_frames = uniform_sample(all_frames, max_num_frames)

    return all_frames




VIDEO_INFO_CACHE = {}


def inference(video_path, prompt, model, processor, shuffle_frame=False, max_num_frames=10, max_new_tokens=25, device="cuda:0", no_video=False, text_only_model=False, extra_video_paths=[], combine_type='target_first'):
   
    

    content = list()
    if not no_video:



        frames = encode_video(video_path, extra_video_paths, combine_type='target_first', max_num_frames=max_num_frames, )
        if shuffle_frame:
            random.shuffle(frames)
            print("frame shuffled")
        content.append(
            {"type": "video", "video": frames,}
        )

        # num_videos = len(video_paths)
        # token_per_video = max_num_frames // num_videos

        # for video_path_single in video_paths:
        # content.append(
        #     {"type": "video", "video": frames, "min_pixels": token_per_video*28*28, "total_pixels": token_per_video*28*28, 'fps': 1}
        # )

        
    content.append(
        {"type": "text", "text": prompt}
    )
    messages = [
        {"role": "user", "content": content},
    ]
    # print(messages)
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    

    inputs = inputs.to(device)
    input_ids = inputs.input_ids
   

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, )
        
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]
    
   
    






def setup_model(model_base, device, text_only_model=False):


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_base, num_crops=4,trust_remote_code=True)
  
    return model, processor


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

def process_work_items(
    work_items,
    model_base,
    device,
    result_dir,
    max_num_frames,
    shuffle_video=False,
    shuffle_frame=False,
    no_video=False,
    num_gpus=1,
    cur_id=0,
    combine_type='target_first',
    custom_question=None,
    add_extra_options=False,
    no_target_video=False,
    replace_correct_with_extra=False,
):
    model, processor = setup_model(model_base, device)
    
    opts = ""

    if shuffle_video:
        opts += "_shuffle_video"
    if shuffle_frame:
        opts += "_shuffle_frame"
    if no_video:
        opts += "_no_video"
    if combine_type:
        if len(work_items[0]['extra_video_path']) != 0:
            opts += f"_combine_type_{combine_type}_num_extra_video_{len(work_items[0]['extra_video_path'])}"
    if custom_question:
        opts += f"_custom_question_{custom_question}"
    if add_extra_options:
        opts += "_add_extra_options"
    if no_target_video:
        opts += "_no_target_video"
    if replace_correct_with_extra:
        opts += "_replace_correct_with_extra"
    
    os.makedirs(f"{result_dir}/{model_base.replace('/', '-')}{opts}", exist_ok=True)
    log_path = f"{result_dir}/{model_base.replace('/', '-')}{opts}/{cur_id}.jsonl"

    
    pbar = tqdm(work_items)
    for idx, item in enumerate(pbar):


        
        video_path = item['video_path']
        if no_target_video:
            video_path = item['extra_video_path'][0]

        # print('custom_question', custom_question, 'add_extra_options', add_extra_options, 'replace_correct_with_extra', replace_correct_with_extra)

        if custom_question is None:
            question = item["question"]
            options = item["options"]
            answer = item["answer"]

        if custom_question == 'video_position':
            option_num = ord(item['answer']) - ord('A') 

            question = f"Which part of the video is most relevant to the given term: {item['options'][option_num]}?\n"
            options = ["A.beginning", "B.middle", "C.end"]
            answer = "A" if combine_type == 'target_first' else "C" if combine_type == 'target_last' else "B" if combine_type == 'random' else "A"
            # example_prompt = GQA_TEMPLATE.replace("[QUESTION]", question)
            # prompt = example_prompt.replace("[OPTION]", str(options))
        
        # if custom_question == 'video_number':
        #     question = f"The given video is combined by multiple videos. How many videos are combined?\n"
        #     options = ["A.1", "B.5", "C.7", "D.3"]
        #     answer = "B" 


        if custom_question == 'static_video_count_frame':
            question = f"The given video is static. How many frames are in the video?\n"
            options = ["A.1", f"B.{max_num_frames}", "C.7", "D.3"]
            answer = "B" 

        if custom_question == 'static_video_bool':
            question = f"Is the given video frozen? \n"
            options = ["A.True", "B.False"]
            answer = "A" 


        if replace_correct_with_extra:
            
            option_num = ord(item['answer']) - ord('A') 
            opt_let = item['answer']
            item['options'][option_num] =  f'{opt_let}.None of the above'
            options = item['options']
           
        


        
        # assert not (custom_question and custom_question), "custom_question and replace_correct_with_extra cannot be True at the same time"
        assert not (add_extra_options and replace_correct_with_extra), "add_extra_options and replace_correct_with_extra cannot be True at the same time"
        
        if add_extra_options:
            option_letter = chr(64 + len(options)) if len(options) <= 2 else chr(96 + len(options))
            options = options + [f'{option_letter.upper()}. I am not sure']
            if no_target_video:
                answer = option_letter.upper()
        
    


        extra_video_paths = item['extra_video_path'] 

        example_prompt = GQA_TEMPLATE.replace("[QUESTION]", question)
        prompt = example_prompt.replace("[OPTION]", str(options))

        





        accs = []
        ious = []

        os.environ["current_sample"] = f"{idx}"

  

        ans = inference(video_path, prompt, model, processor, shuffle_frame=shuffle_frame, max_num_frames=max_num_frames, device=device, extra_video_paths=extra_video_paths, combine_type='target_first')

        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = re.search(pattern_answer, ans, re.DOTALL)


    

        acc = 0.0
        if match_answer:
            answer = match_answer.group(1)
            if extract_characters_regex(answer) == extract_characters_regex(answer):
                acc = 1.0

        if answer in ans:  
            acc = 1.0

        accs.append(acc)

        
        
        item_res = {'video_path': video_path, 'prompt':prompt, 'gt':answer, 'pred':ans, 'acc':acc }
        append_to_jsonl(log_path, item_res)
        
        pbar.set_postfix({'accuracy': sum(accs)/len(accs), 'gpu_id': cur_id})
        
        # except Exception as e:
        #     print(f"Error processing {video_path}: {e}")
        

def evaluate(dataset, args):
    
    process_work_items(
        work_items=dataset, 
        model_base=args.model_base, 
        device=f'cuda:{args.cur_gpu}', 
        result_dir=f'{args.result_dir}',
        max_num_frames=args.max_num_frames,
        shuffle_video=args.shuffle_video,
        shuffle_frame=args.shuffle_frame,
        no_video=args.no_video,
        num_gpus=args.num_gpus,
        cur_id=args.cur_gpu,
        combine_type=args.combine_type,
        custom_question=args.custom_question,
        add_extra_options=args.add_extra_options,
        no_target_video=args.no_target_video,
        replace_correct_with_extra=args.replace_correct_with_extra,
    )

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for nextgqa')
    parser.add_argument('--dataset_name', default='videomme', type=str, help='Specify the dataset.')
    parser.add_argument('--video_root', default='', type=str, help='Specify the video root.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/path/to/qwen-model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--result_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPU device to use")
    parser.add_argument("--cur_gpu", type=int, default=0, help="Current GPU device")
    parser.add_argument("--no_video",  action="store_true", default=False, help="Not passing video to model")
    parser.add_argument("--shuffle_video", action="store_true", default=False, help="Shuffle video")
    parser.add_argument("--shuffle_frame", action="store_true", default=False, help="Shuffle frame")
    parser.add_argument("--limit", type=float, default=1,  help="dataset size")
    parser.add_argument("--combine_type", type=str, default='target_first', help="combine type")
    parser.add_argument("--custom_question", type=str, default=None, help="custom question")
    parser.add_argument("--add_extra_options", action="store_true", help="add extra options")
    parser.add_argument("--no_target_video", action="store_true", help="no target video")
    parser.add_argument("--replace_correct_with_extra", action="store_true", help="replace correct with extra video")
    parser.add_argument("--num_extra_video", type=int, default=0, help="number of extra video")
    parser.add_argument("--max_num_frames", type=int, default=16, help="max number of frames")
    parser.add_argument("--data_files", type=str, default=None, help="data files")
    return parser.parse_args()


if __name__=='__main__':
    print("Starting Qwen2.5-VL evaluation")
    args = get_args()
    print(f"args: {args}")
    dataset = build_data(args.dataset_name, args.video_root, args.data_files, shuffle_video=args.shuffle_video, num_gpus=args.num_gpus, cur_gpu=args.cur_gpu, limit=args.limit, num_extra_video=args.num_extra_video)

    print(f"dataset_name: {args.dataset_name}")
    num_gpus = args.num_gpus
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus
    
    evaluate(dataset, args)