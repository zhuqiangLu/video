
from transformers import AutoProcessor, AutoModelForImageTextToText
from .builder import modeling_funcs_builder
import torch
import numpy as np



def dummy():
    pass



def get_inputs_func(prompt, frames, processor, no_video=False, video_path=None, extra_video_paths=None):

    content = list()
    content.append({"type": "text", "text": prompt})

    

    if not no_video:
        # we use dummy video to get video placeholder 
        content.append({"type": "video", "path": "./asset/test.mp4"})
    else:
        print("no video")
    messages = [
        {"role": "user", "content": content},
    ]
    

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=len(frames),
    ).to(dtype=torch.bfloat16)
    
   
    inputs.pixel_values = processor.video_processor(frames, return_tensors="pt").pixel_values.to(dtype=torch.bfloat16)
    # pixel_values_test = torch.cat(pixel_values_test, dim=0).to(dtype=torch.bfloat16)


    # print(inputs.input_ids.shape, inputs.pixel_values.shape, pixel_values_test.shape)
    # raise


    return inputs



def setup_model(model_base, device, text_only_model=False):
    print(f"setup_model {model_base}")

    
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


modeling_funcs_builder.register("SmolVLM2", setup_model, get_inputs_func)