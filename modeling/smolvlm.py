
from transformers import AutoProcessor, AutoModelForImageTextToText
from .builder import modeling_funcs_builder
import torch
import numpy as np



def dummy():
    pass



def get_inputs_func(prompt, frames, processor,  ppl=False, answer=None):

    content = list()
    content.append({"type": "text", "text": prompt})

    

    # we use dummy video to get video placeholder 
    if len(frames) > 0:
        content.append({"type": "video", "path": "./asset/test.mp4"})
    
    messages = [
        {"role": "user", "content": content},
    ]
    
    ppl_inputs = dict() 

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=not ppl,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=len(frames),
    ).to(dtype=torch.bfloat16)
    
    
    
    if ppl:
        assert answer is not None
        start_idx = inputs.input_ids.shape[1] 

        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    num_frames=len(frames),
                ).to(dtype=torch.bfloat16)

        ppl_inputs['start_idx'] = start_idx

    


    if len(frames) > 0:
        inputs.pixel_values = processor.video_processor(frames, return_tensors="pt").pixel_values.to(dtype=torch.bfloat16)

    
    return inputs, ppl_inputs 
  



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