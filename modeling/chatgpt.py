from .builder import modeling_funcs_builder
# from IPython.display import display, Image, Audio

# import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import base64
import io
# from dataloader import build_data

class ChatGPT:
    def __init__(self, model_base):
        self.model_base = model_base.split('/')[-1]
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

    def generate(self, prompt, frames, processor):
        
        response = self.client.responses.create(
            model=self.model_base,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                prompt
                            )
                        },
                        *[
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{frame}"
                            }
                            for frame in frames
                        ]
                    ]
                }
            ],
        )

        return response.output_text

        

def dummy():
    pass

def PIL_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")  # Choose format (JPEG/PNG/etc.)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def get_inputs_func(prompt, frames, processor):
    base64Frames = list()
    # Convert to Base64

    base64Frames = [PIL_to_base64(frame) for frame in frames]

    ret = dict()
    ret['prompt'] = prompt
    ret['base64Frames'] = base64Frames

    return ret
    
def inference_func(model, processor, inputs, max_new_tokens=20, use_cache=True):

    prompt = inputs['prompt']
    base64Frames = inputs['base64Frames']
    response = model.generate(prompt, base64Frames, None)

    return response



def setup_model(model_base, device):
    print(f"warning: setting up chatgpt client, make sure model_base is in this form 'openai/gpt-4o-mini', model_base should be model version, please make sure you have set the OPENAI_API_KEY in the environment variables")
    model = ChatGPT(model_base)
  
    return model, None

modeling_funcs_builder.register("openai", setup_model, get_inputs_func, inference_func)

