from transformers import AutoProcessor, AutoModelForImageTextToText
from .builder import modeling_funcs_builder
import torch
# from dataloader import build_data


def dummy():
    pass

def get_inputs_func(prompt, frames, processor):

    content = list()
    content.append({"type": "text", "text": prompt})

    

    for image in frames:
        content.append({"type": "image", "image": image})
    
    messages = [
        {"role": "user", "content": content},
    ]

    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=frames, padding=True, return_tensors="pt")


    
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

modeling_funcs_builder.register("gemma-3n", setup_model, get_inputs_func)

