
from transformers import AutoProcessor, AutoModelForCausalLM
from .builder import modeling_funcs_builder
import torch



def dummy():
    pass



def get_inputs_func(prompt, frames, processor, ppl=False, answer=None):

    placeholder = "" 
    for idx, _ in enumerate(frames):
        placeholder += f"<|image_{idx+1}|>\n"
    content = placeholder+prompt
    messages = [
        {"role": "user", "content": content},
    ]

    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not ppl)
    inputs = processor(text=text, images=frames if len(frames) > 0 else None, padding=False, return_tensors="pt")

    ppl_inputs = dict()

    if ppl:
        assert answer is not None
        start_idx = inputs.input_ids.shape[1] 

        messages.append({"role": "assistant", "content": answer})
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=text, images=frames if len(frames) > 0 else None, padding=False, return_tensors="pt")
        ppl_inputs['start_idx'] = start_idx


    
    return inputs, ppl_inputs 




def setup_model(model_base, device, text_only_model=False):

    print(f"setup_model {model_base}")
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