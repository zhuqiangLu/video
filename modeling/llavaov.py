
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from .builder import modeling_funcs_builder
import torch
import numpy as np



def dummy():
    pass



def get_inputs_func(prompt, frames, processor, no_video=False):

    content = list()
    content.append({"type": "text", "text": prompt})

    

    if not no_video:
        # we use dummy video to get video placeholder 
        # content.append({"type": "video", "path": "./asset/test.mp4"})
        for _ in frames:
            content.append({"type": "image"})
    else:
        print("no video")
    messages = [
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=frames, text=prompt, return_tensors='pt').to(torch.bfloat16)
    # print(inputs.keys())
    # raise
    
   
    # inputs.pixel_values = processor.video_processor(frames, return_tensors="pt").pixel_values.to(dtype=torch.bfloat16)
    

    return inputs



def setup_model(model_base, device, text_only_model=False):

    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_base, num_crops=4,trust_remote_code=True)
  
    return model, processor


modeling_funcs_builder.register("onevision", setup_model, get_inputs_func)