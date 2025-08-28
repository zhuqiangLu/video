import json
import re
from decord import VideoReader, cpu
from PIL import Image
import random
# from GQA_TEMPLATE import GQA_TEMPLATE
from tqdm import tqdm 
import os
from dataloader import data_builder
import torch


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
    for video_path in video_paths:

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

def get_frames(video_path, extra_video_paths, frozen_video=False, combine_type=None, shuffle_frame=False, max_num_frames=10):
    frames = encode_video(video_path, extra_video_paths, combine_type=combine_type, max_num_frames=max_num_frames,)
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
    decode_func,
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
    if os.path.exists(log_path):
        os.remove(log_path)

    
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

        if custom_question == 'frozen_video_count_frame':
            question = f"The given video is static. How many frames are in the video?\n"
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
            options = options + [f'{option_letter.upper()}. I am not sure']
            if no_target_video:
                answer = option_letter.upper()
        
    


        extra_video_paths = item['extra_video_path'] 

        example_prompt = GQA_TEMPLATE.replace("[QUESTION]", question)
        prompt = example_prompt.replace("[OPTION]", str(options))

    

        accs = []
  

        pred = inference(video_path,
                         get_inputs_func, 
                         decode_func, 
                         prompt, 
                         model, 
                         processor, 
                         shuffle_frame=shuffle_frame, 
                         frozen_video=frozen_video,
                         no_video=no_video, 
                         max_num_frames=max_num_frames, 
                         max_new_tokens=max_new_tokens,
                         device=device, 
                         extra_video_paths=extra_video_paths, 
                         combine_type=combine_type, 
                         custom_question=custom_question, 
                         )

        acc = 0.0
        if extract_characters_regex(answer) == extract_characters_regex(pred):
            acc = 1.0

        if answer in pred[:1]:  
            acc = 1.0

        accs.append(acc)

        
        
        item_res = {'video_path': video_path, 'prompt':prompt, 'gt':answer, 'pred':pred, 'acc':acc }
        append_to_jsonl(log_path, item_res)
        
        pbar.set_postfix({'accuracy': sum(accs)/len(accs), "gpu_id": cur_id})
        
   




def inference(video_path, 
              get_inputs_func,
              decode_func,
              prompt, 
              model, 
              processor, 
              shuffle_frame=False, 
              frozen_video=False,
              no_video=False, 
              max_num_frames=10, 
              max_new_tokens=20, 
              device="cuda:0", 
              extra_video_paths=[], 
              combine_type=None, 
              custom_question=None,
              ):
   
    
    frames = get_frames(video_path, extra_video_paths, frozen_video=frozen_video, combine_type=combine_type, shuffle_frame=shuffle_frame, max_num_frames=max_num_frames)
    
    inputs = get_inputs_func(prompt, frames, processor, no_video=no_video)

   
    inputs = inputs.to(device)
    input_ids = inputs.input_ids
   

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, )




    if decode_func is None:
        generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
    else:
        output_text = decode_func(output_ids, inputs, processor)
        return output_text

    