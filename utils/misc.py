import json
import re
from decord import VideoReader, cpu
from PIL import Image
import random
# from GQA_TEMPLATE import GQA_TEMPLATE
from tqdm import tqdm 
import os
from dataloader import build_data


GQA_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION]. Only output the corresponding letter of the option.
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





def encode_video(target_video_path, extra_video_paths, max_num_frames=10, combine_type='target_first'):
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

def get_frames(video_path, extra_video_paths, combine_type='target_first', max_num_frames=10):
    frames = encode_video(video_path, extra_video_paths, combine_type=combine_type, max_num_frames=max_num_frames)
    if 'static' in custom_question:
        # Get a random frame and duplicate it
        random_frame = random.choice(frames)
        frames = [random_frame] * max_num_frames
        print("static video")

    if shuffle_frame:
        random.shuffle(frames)
        print("frame shuffled")
    return frames 


def get_dataset(args):
    dataset = build_data(args.dataset_name, args.video_root, args.data_files, shuffle_video=args.shuffle_video, num_gpus=args.num_gpus, cur_gpu=args.cur_gpu, limit=args.limit, num_extra_video=args.num_extra_video)
    return dataset


def run_experiment(
    inference_func,
    dataset,
    model_base,
    device,
    result_dir,
    max_num_frames,
    shuffle_frame=False,
    shuffle_video=False,
    no_video=False,
    num_gpus=1,
    cur_id=0,
    combine_type=None,
    custom_question=None,
    add_extra_options=False,
    no_target_video=False,
    replace_correct_with_extra=False,
):
    model, processor = setup_model(model_base, device)
    
    '''
       Manage the saved file name 
    '''
    opts = ""

    if shuffle_video:
        opts += "_shuffle_video"
    if no_video:
        opts += "_no_video"
    if combine_type:
        if len(dataset[0]['extra_video_path']) != 0:
            opts += f"_combine_type_{combine_type}_num_extra_video_{len(dataset[0]['extra_video_path'])}"
    if custom_question:
        opts += f"_custom_question_{custom_question}"
    if add_extra_options:
        opts += "_add_extra_options"
    if no_target_video:
        opts += "_no_target_video"
    if replace_correct_with_extra:
        opts += "_replace_correct_with_extra"
    if shuffle_frame:
        opts += "_shuffle_frame"
    
    os.makedirs(f"{result_dir}/{model_base.replace('/', '-')}{opts}", exist_ok=True)
    log_path = f"{result_dir}/{model_base.replace('/', '-')}{opts}/{cur_id}.jsonl"

    
    pbar = tqdm(dataset)
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
        
        if custom_question == 'video_number':
            question = f"The given video is combined by multiple videos. How many videos are combined?\n"
            options = ["A.1", f"B.{len(item['extra_video_path'])+1}", "C.7", "D.3"]
            answer = "B" 

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
  

        pred = inference_func(video_path, prompt, model, processor, shuffle_frame=shuffle_frame, max_num_frames=max_num_frames, device=device, extra_video_paths=extra_video_paths, combine_type=combine_type, custom_question=custom_question)

    
        if extract_characters_regex(answer) == extract_characters_regex(pred):
            acc = 1.0

        if answer in pred[:1]:  
            acc = 1.0

        accs.append(acc)

        
        
        item_res = {'video_path': video_path, 'prompt':prompt, 'gt':answer, 'pred':ans, 'acc':acc }
        append_to_jsonl(log_path, item_res)
        
        pbar.set_postfix({'accuracy': sum(accs)/len(accs), "gpu_id": cur_id})
        
   