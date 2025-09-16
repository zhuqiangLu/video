
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .builder import modeling_funcs_builder
import torch


from qwen_vl_utils import process_vision_info

def dummy():
    pass
    
def get_inputs_func(prompt, frames, processor,  ppl=False, answer=None):    
    content = list()
    if len(frames) > 0:
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
    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    ppl_inputs = dict() 

    if ppl:
        assert answer is not None
        start_idx = inputs.input_ids.shape[1] 

        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        ppl_inputs['start_idx'] = start_idx

    return inputs, ppl_inputs 
    



def setup_model(model_base, device, text_only_model=False):
    print(f"setup_model {model_base}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_base, num_crops=4,trust_remote_code=True)
  
    return model, processor

modeling_funcs_builder.register("Qwen2.5-VL", setup_model, get_inputs_func)
    