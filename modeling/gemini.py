from .builder import modeling_funcs_builder
# from IPython.display import display, Image, Audio

# import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import time
from google import genai
from google.genai import types
import os, tempfile
import io
import numpy as np
import cv2
from PIL import Image
# from dataloader import build_data

class Gemini:
    def __init__(self, model_base):
        self.model_base = f"models/{model_base.split('/')[-1]}"
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "<your GEMINI API key if not set as env var>"))

    def generate(self, prompt, frames, processor):
        
        response = self.client.models.generate_content(
            model=self.model_base,
            contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=frames, mime_type='video/mp4')
                ),
                types.Part(text=prompt)
            ]
        )
            )
        return response.text    

        

def dummy():
    pass


def pil_images_to_video_bytes(pil_images, fps=1, size=None, fourcc="mp4v"):
    """
    Turn a list of PIL.Image into MP4 bytes reliably by writing to a temp file first.
    """
    if not pil_images:
        raise ValueError("pil_images is empty")

    # Normalize size (W, H)
    if size is None:
        size = pil_images[0].size
    w, h = size

    # Create temp .mp4 path
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*fourcc),  # "mp4v" usually available
            float(fps),
            (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open cv2.VideoWriter; missing codec support?")

        for img in pil_images:
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
            if img.mode != "RGB":
                img = img.convert("RGB")
            frame = np.array(img)                       # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def get_inputs_func(prompt, frames, processor, no_video=False):

    
    # Convert to Base64

    
    video_bytes = None
    if not no_video:
        video_bytes = pil_images_to_video_bytes(frames)
        # video_bytes = open('asset/test.mp4', 'rb').read()

    ret = dict()
    ret['prompt'] = prompt
    ret['video_bytes'] = video_bytes

    return ret
    
def inference_func(model, processor, inputs, max_new_tokens=20, use_cache=True):

    prompt = inputs['prompt']
    video_bytes = inputs['video_bytes']
    response = model.generate(prompt, video_bytes, None)

    return response



def setup_model(model_base, device):
    print(f"warning: setting up gemini client, make sure model_base is in this form 'google/gemini-2.5-flash', model_base should be model version, please make sure you have set the GEMINI_API_KEY in the environment variables")
    model = Gemini(model_base)
  
    return model, None

modeling_funcs_builder.register("google", setup_model, get_inputs_func, inference_func)

