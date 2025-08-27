from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils.misc import get_frames, run_experiment, get_dataset
from utils.args import get_args
# from dataloader import build_data




def inference(video_path, 
              prompt, 
              model, 
              processor, 
              shuffle_frame=False, 
              max_num_frames=10, 
              max_new_tokens=20, 
              device="cuda:0", 
              no_video=False, 
              text_only_model=False, 
              extra_video_paths=[], 
              combine_type=None, 
              custom_question=None,
              ):
   
    content = list()
    content.append({"type": "text", "text": prompt})
    frames = get_frames(video_path, extra_video_paths, combine_type=combine_type, max_num_frames=max_num_frames)
    
    if not no_video:
        for image in frames:
            content.append({"type": "image", "image": image})
    messages = [
        {"role": "user", "content": content},
    ]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=text, images=frames, padding=True, return_tensors="pt")
    
    
    inputs = inputs.to(device)
    input_ids = inputs.input_ids
   

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, )
        
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]
    
   
    

def setup_model(model_base, device, text_only_model=False):

    
    model = AutoModelForImageTextToText.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_base, num_crops=4,trust_remote_code=True)
  
    return model, processor

     


if __name__=='__main__':
    args = get_args()
    # dataset = build_data(args.dataset_name, args.video_root, args.data_files,  shuffle_video=args.shuffle_video, num_gpus=args.num_gpus, cur_gpu=args.cur_gpu, limit=args.limit, num_extra_video=args.num_extra_video)
    dataset = get_dataset(args)
    print(f"dataset_name: {args.dataset_name}")
    num_gpus = args.num_gpus
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus
    
    evaluate(dataset, args)


    run_experiment(
        inference_func=inference,
        work_items=dataset, 
        
        model_base=args.model_base, 
        device=f'cuda:{args.cur_gpu}', 
        result_dir=args.result_dir,
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
        max_num_frames=args.max_num_frames,
    )
