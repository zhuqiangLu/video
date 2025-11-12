
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
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
    # content.append({"type": "video", "path": "./asset/test.mp4"})
    for _ in frames:
        content.append({"type": "image"})
    
    messages_user = [
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(messages_user, add_generation_prompt=True)

    frames = None if len(frames) == 0 else frames

    # resize frames 
    if frames is not None:
        frames = [frame.resize((336, 336)) for frame in frames]

    inputs = processor(images=frames, text=prompt, return_tensors='pt').to(torch.bfloat16)
    ppl_inputs = dict() 
    if ppl:
        assert answer is not None
        # 1) User + assistant header (empty assistant text)
        msgs_head = messages_user + [{"role": "assistant", "content": [{"type":"text","text":""}]}]
        text_head = processor.apply_chat_template(msgs_head, tokenize=False, add_generation_prompt=False)
        enc_head  = processor(text=text_head, images=frames, padding=False, return_tensors="pt")

        # 2) Full = User + assistant full answer
        msgs_full = messages_user + [{"role": "assistant", "content": [{"type":"text","text":answer}]}]
        text_full = processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        enc_full  = processor(text=text_full, images=frames, padding=True, return_tensors="pt")

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

    return inputs, ppl_inputs 
   



def setup_model(model_base, device, text_only_model=False):

    print(f"setup_model {model_base}")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_base, trust_remote_code=True)
  
    return model, processor


modeling_funcs_builder.register("onevision", setup_model, get_inputs_func)