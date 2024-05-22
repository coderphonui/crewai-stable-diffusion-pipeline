from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool
from typing import Type, Any
from datetime import datetime
import urllib.request
import base64
import json
import time
import os


# TODO - enhance more
# Set checkpoint: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/7839
# Upscale: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/6874
webui_server_url = 'http://127.0.0.1:7860'

out_dir = 'output'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
os.makedirs(out_dir_t2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(**payload):
    print(payload)
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)
        return save_path



class StableDiffusionToolSchema(BaseModel):
    """Input for StableDiffusionTool."""
    prompt: str = Field(..., description="Mandatory prompt to generate the photo")
    negative_prompt: str = Field(..., description="Negative prompt")

    

class StableDiffusionTool(BaseTool):
    name: str = "StableDiffusionGenerator"
    description: str = "Create image from Stable Diffusion with input prompt and negative prompt"
    args_schema: Type[BaseModel] = StableDiffusionToolSchema

    def _run(self, **kwargs: Any) -> str:
        prompt = kwargs.get('prompt')
        negative_prompt = kwargs.get('negative_prompt')
        payload = {
            "prompt": prompt, 
            "negative_prompt": negative_prompt,
            "seed": 1,
            "steps": 40,
            "width": 768,
            "height": 768,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M",
            "n_iter": 1,
            "batch_size": 1,
        }
        return call_txt2img_api(**payload)