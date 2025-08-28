
from transformers import AutoProcessor, AutoModelForCausalLM
from .builder import modeling_funcs_builder
import torch



def dummy():
    pass



def get_inputs_func(prompt, frames, processor, no_video=False):

    placeholder = "" 
    if not no_video:
        for idx, _ in enumerate(frames):
            placeholder += f"<|image_{idx+1}|>\n"
    content = placeholder+prompt
    messages = [
        {"role": "user", "content": content},
    ]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=text, images=frames, padding=True, return_tensors="pt")
    return inputs




def setup_model(model_base, device, text_only_model=False):

    
    model = AutoModelForCausalLM.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_base, num_crops=4,trust_remote_code=True)
  
    return model, processor


modeling_funcs_builder.register("Phi-3.5-vision-instruct", setup_model, get_inputs_func)