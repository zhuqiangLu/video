
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .builder import modeling_funcs_builder
import torch


from qwen_vl_utils import process_vision_info
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
    messages_user = [
        {"role": "user", "content": content},
    ]
    # print(messages)
    # Preparation for inference
    text = processor.apply_chat_template(
        messages_user, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info([messages_user])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )
    ppl_inputs = dict() 

    if ppl:
        assert answer is not None
        # 1) User + assistant header (empty assistant text)
        msgs_head = messages_user + [{"role": "assistant", "content": [{"type":"text","text":""}]}]
        text_head = processor.apply_chat_template(msgs_head, tokenize=False, add_generation_prompt=False)
        enc_head  = processor(text=text_head, images=image_inputs, padding=False, return_tensors="pt")

        # 2) Full = User + assistant full answer
        msgs_full = messages_user + [{"role": "assistant", "content": [{"type":"text","text":answer}]}]
        text_full = processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        enc_full  = processor(text=text_full, images=image_inputs, padding=True, return_tensors="pt")

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



        # start_idx = inputs.input_ids.shape[1] 

        # messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # # print(text)
        # inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=False, return_tensors="pt")
        
        # # raise

        # ppl_inputs['start_idx'] = start_idx


   
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
    min_pixels = 256 * 28 * 28
    max_pixels = 10240 * 28 * 28 
    from transformers import AutoImageProcessor

    processor = AutoProcessor.from_pretrained(model_base,  min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
  
    return model, processor

modeling_funcs_builder.register("Qwen2.5-VL", setup_model, get_inputs_func)
modeling_funcs_builder.register("Video-R1", setup_model, get_inputs_func)
    
