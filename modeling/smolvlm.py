
from transformers import AutoProcessor, AutoModelForImageTextToText
from .builder import modeling_funcs_builder
import torch
import numpy as np



def dummy():
    pass

def _insertion_span(full_ids, head_ids):
    """Find smallest [start:end) in full such that removing it yields head."""
    f = full_ids.tolist()
    h = head_ids.tolist()
    i = 0
    # advance from left while equal
    while i < len(f) and i < len(h) and f[i] == h[i]:
        i += 1
    # advance from right while equal
    k = 0
    while k < (len(f) - i) and k < (len(h) - i) and f[-1 - k] == h[-1 - k]:
        k += 1
    start = i
    end = len(f) - k
    return start, end

def get_inputs_func(prompt, frames, processor,  ppl=False, answer=None):

    content = list()
    content.append({"type": "text", "text": prompt})

    

    # we use dummy video to get video placeholder 
    if len(frames) > 0:
        content.append({"type": "video", "path": "./asset/test.mp4"})
    
    messages_user = [
        {"role": "user", "content": content},
    ]
    
    ppl_inputs = dict() 

    inputs = processor.apply_chat_template(
        messages_user,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=len(frames),
    ).to(dtype=torch.bfloat16)
    
    
    
    if ppl:
        assert answer is not None
        start_idx = inputs.input_ids.shape[1] 


        # 1) User + assistant header (empty assistant text)
        msgs_head = messages_user + [{"role": "assistant", "content": [{"type":"text","text":""}]}]
        enc_head = processor.apply_chat_template(msgs_head, tokenize=True, add_generation_prompt=False, num_frames=len(frames), padding=False, return_tensors="pt", return_dict=True,).to(dtype=torch.bfloat16)
        # enc_head  = processor(text=text_head, images=frames, padding=False, return_tensors="pt", num_frames=len(frames))

        # 2) Full = User + assistant full answer
        msgs_full = messages_user + [{"role": "assistant", "content": [{"type":"text","text":answer}]}]
        enc_full = processor.apply_chat_template(msgs_full, tokenize=True, add_generation_prompt=False, num_frames=len(frames), padding=True, return_tensors="pt", return_dict=True,).to(dtype=torch.bfloat16)
        # enc_full  = processor(text=text_full, images=frames, padding=True, return_tensors="pt", num_frames=len(frames))

        inputs = enc_full

        full_ids = enc_full.input_ids
        head_ids = enc_head.input_ids
        attn     = enc_full.attention_mask if "attention_mask" in enc_full else torch.ones_like(full_ids)

        # Find the exact inserted subsequence for the assistant text
        start_idx, end_idx = _insertion_span(full_ids[0], head_ids[0])

        # Build labels = only assistant text tokens; ignore padding/specials
        labels = full_ids.clone()
        labels[:, :start_idx] = -100
        labels[:, end_idx:]   = -100
        labels[attn == 0]     = -100

        ppl_inputs.update({
                "labels": labels,
                "start_idx": start_idx,
                "end_idx": end_idx,
            })

            

        # messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        # inputs = processor.apply_chat_template(
        #             messages,
        #             add_generation_prompt=False,
        #             tokenize=True,
        #             return_dict=True,
        #             return_tensors="pt",
        #             num_frames=len(frames),
        #         ).to(dtype=torch.bfloat16)

        # ppl_inputs['start_idx'] = start_idx

    


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