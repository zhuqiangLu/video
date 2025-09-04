import json
import re
try:
    from decord import VideoReader, cpu
except:
    print("decord not installed")

try:
    import av
except:
    print("av not installed")

from PIL import Image
import random
# from GQA_TEMPLATE import GQA_TEMPLATE
from tqdm import tqdm 
import os
from dataloader import data_builder
import torch
import numpy as np

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



def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def encode_video(target_video_path, extra_video_paths, max_num_frames=10, combine_type='target_first', backend='av', start_time=None, end_time=None):
    
    
    if combine_type == 'target_first':
        print(f'target_first with num video: {len(extra_video_paths)}')
        video_paths = [target_video_path, *extra_video_paths]
    elif combine_type == 'target_last':
        print(f'target_last with num video: {len(extra_video_paths)}')
        video_paths = [*extra_video_paths, target_video_path]
    elif combine_type == 'target_middle':
        print(f'target_middle with num video: {len(extra_video_paths)}')
        video_paths = [*extra_video_paths[:1], target_video_path, *extra_video_paths[1:]]
        random.shuffle(video_paths)
    else:
        video_paths = [target_video_path]
        print(f'no combine with num video: {len(extra_video_paths)} for option {combine_type}, do nothing')
        
    
    all_frames = []
    for video_path in video_paths:
        if backend == 'decord':
            vr = VideoReader(video_path, ctx=cpu(0))
            print(vr)
            sample_fps = round(vr.get_avg_fps() / 1)  # FPS
            frame_idx = [i for i in range(0, len(vr), sample_fps)]
            if len(frame_idx) > max_num_frames:
                frame_idx = uniform_sample(frame_idx, max_num_frames)
            frames = vr.get_batch(frame_idx).asnumpy()
            frames = [Image.fromarray(v.astype('uint8')) for v in frames]
            all_frames.extend(frames) 
        elif backend == 'av':
            frames = sample_frames(video_path, max_num_frames, start_time, end_time)
            all_frames.extend(frames) 



    # all_frames = uniform_sample(all_frames, max_num_frames)

    return all_frames


def sample_frames(video_path, max_num_frames, start_time, end_time, ):
    """
    Sample frames between start_time and end_time at a given fps.
    
    Args:
        video_path (str): Path to video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        fps (float): Desired sampling fps.
        
    Returns:
        List of (timestamp, frame_ndarray).
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Seek close to the start time (in pts units)
    if start_time is not None:
        container.seek(int(start_time / video_stream.time_base))
        next_sample_time = start_time
    else:
        next_sample_time = 0 


    frames = []

    # sample_fps = round(container.streams.video[0].average_rate / 1)  # FPS
    # step = 1.0 / fps  # interval between samples (seconds)
    step = 1.0 # sample at 1 fps

    for frame in container.decode(video_stream):
        timestamp = frame.pts * video_stream.time_base

        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            break


        if timestamp >= next_sample_time:
            img = frame.to_image()
            frames.append(img)
            next_sample_time += step

    duration = float(container.duration/ av.time_base)
    container.close()
    print(f'sample {len(frames)} frames from video with duration {duration:.2f}s from {video_path}, start_time {start_time}, end_time {end_time}')
    if len(frames) > max_num_frames:
        # Uniformly sample frames to reduce to max_num_frames
        indices = list(range(len(frames)))
        sample_indices = uniform_sample(indices, max_num_frames)
        frames = [frames[i] for i in sample_indices]
    return frames    

def get_frames_by_indices_pyav(video_path, max_num_frames):
    container = av.open(video_path)
    sample_fps = round(container.streams.video[0].average_rate / 1)  # FPS
    num_frames = container.streams.video[0].frames
    frame_idx = [i for i in range(0, num_frames, sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    # print(container.streams.video[0].average_rate, frame_idx, num_frames, video_path,)
    # raise
    stream = container.streams.video[0]
    result = list()
    for frame_i, frame in enumerate(container.decode(stream)):
        if frame_i in frame_idx:
            result.append(frame.to_image())
            # print(type(frame.to_image()))
        # if frame_i > max(indices):
        #     break
    # if len(result) != max_num_frames:
    #     print(f'warning: trying to sample {max_num_frames} frames from {video_path}, but got {len(result)} frames')
    return result

def get_frames(video_path, extra_video_paths, frozen_video=False, combine_type=None, shuffle_frame=False, max_num_frames=10, start_time=None, end_time=None):
    frames = encode_video(video_path, extra_video_paths, combine_type=combine_type, max_num_frames=max_num_frames, start_time=start_time, end_time=end_time)
    
    if len(extra_video_paths) > 0:
        # unify frame resolution 
        max_width = max(frame.width for frame in frames)    
        max_height = max(frame.height for frame in frames)
        for idx, frame in enumerate(frames):
            frames[idx] = frame.resize((max_width, max_height))



    if frozen_video:
        # Get a random frame and duplicate it
        random_frame = random.choice(frames)
        frames = [random_frame] * max_num_frames
        print("frozen video")

    if shuffle_frame:
        random.shuffle(frames)
        print("frame shuffled")
    return frames





def run_experiment(
    # inference_func,
    get_inputs_func,
    inference_func,
    dataset,
    model_base,
    setup_model_func,
    device,
    result_dir,
    max_num_frames,
    max_new_tokens,
    shuffle_frame=False,
    shuffle_video=False,
    frozen_video=False,
    no_video=False,
    num_gpus=1,
    cur_id=0,
    combine_type=None,
    custom_question=None,
    add_extra_options=False,
    no_target_video=False,
    replace_correct_with_extra=False,
    resume=False,
):
    model, processor = setup_model_func(model_base, device)
    
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
    if frozen_video:
        opts += "_frozen_video"
    os.makedirs(f"{result_dir}/{model_base.replace('/', '-')}{opts}", exist_ok=True)
    log_path = f"{result_dir}/{model_base.replace('/', '-')}{opts}/{cur_id}.jsonl"

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

        
    

        
        video_path = item['video_path']
        start_time = item['start_time'] if 'start_time' in item else None
        end_time = item['end_time'] if 'end_time' in item else None
        if no_target_video:
            video_path = item['extra_video_path'][0]

        # print(f"processing {video_path}")

        # print('custom_question', custom_question, 'add_extra_options', add_extra_options, 'replace_correct_with_extra', replace_correct_with_extra)

        if custom_question is None:
            question = item["question"]
            options = item["options"]
            answer = item["answer"]

        if custom_question == 'video_position':
            option_num = ord(item['answer']) - ord('A') 

            question = f"Which part of the video is most relevant to the given term: {item['options'][option_num]}?\n"
            options = ["A.beginning", "B.middle", "C.end"]
            if combine_type == 'target_first':
                answer = "A"
            elif combine_type == 'target_last':
                answer = "C"
            elif combine_type == 'target_middle':
                answer = "B"
            else:
                raise ValueError(f"Invalid combine_type: {combine_type}")
            # example_prompt = GQA_TEMPLATE.replace("[QUESTION]", question)
            # prompt = example_prompt.replace("[OPTION]", str(options))
        
        if custom_question == 'video_number':
            question = f"The given video is combined by multiple videos. How many videos are combined?\n"
            correct_num = len(item['extra_video_path'])+1 
            # Generate 3 random numbers from 0-10 excluding correct_num
            possible_nums = list(range(11))
            possible_nums.remove(correct_num)
            wrong_nums = np.random.choice(possible_nums, size=3, replace=False)
            options = [f"A.{wrong_nums[0]}", f"B.{wrong_nums[1]}", f"C.{wrong_nums[2]}", f"D.{correct_num}"]
            answer = "D" 

        if custom_question == 'count_frame':
            question = f"How many frames are in the video?\n"
            options = ["A.1", f"B.{max_num_frames}", "C.7", "D.3"]
            answer = "B" 

        if custom_question == 'frozen_video_bool':
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
            options = options + [f'{option_letter.upper()}. I do not know']
            if no_target_video:
                answer = option_letter.upper()
        
    


        extra_video_paths = item['extra_video_path'] 

        example_prompt = GQA_TEMPLATE.replace("[QUESTION]", question)
        prompt = example_prompt.replace("[OPTION]", str(options))

    

        accs = []
  

        frames = get_frames(video_path, extra_video_paths, frozen_video=frozen_video, combine_type=combine_type, shuffle_frame=shuffle_frame, max_num_frames=max_num_frames, start_time=start_time, end_time=end_time)
        inputs = get_inputs_func(prompt, frames, processor, no_video=no_video)
        continue 
  
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
            
            pbar.set_postfix({'accuracy': sum(accs)/len(accs), "gpu_id": cur_id})

        except Exception as e: 
            print(f"Error: {e}")
            print(f"video_path: {video_path}")


            



    

    