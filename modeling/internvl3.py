
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from .builder import modeling_funcs_builder
import torch


from qwen_vl_utils import process_vision_info

def dummy():
    pass
    
def get_inputs_func(prompt, frames, processor, no_video=False):    
    content = list()
    if not no_video:
        
        content.append(
            {"type": "video", "video": frames,}
        )

        
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
    return inputs
    



def setup_model(model_base, device, text_only_model=False):


    model = AutoModel.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().to(device)

    # processor = AutoProcessor.from_pretrained(model_base, num_crops=4,trust_remote_code=True)
    processor = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True, use_fast=False)

  
    return model, processor

modeling_funcs_builder.register("Qwen2.5-VL", setup_model, get_inputs_func)
    