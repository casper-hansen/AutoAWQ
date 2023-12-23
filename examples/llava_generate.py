import requests
import torch
from PIL import Image

from awq import AutoAWQForCausalLM
from transformers import AutoProcessor

quant_path = "ybelkada/llava-1.5-7b-hf-awq"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, safetensors=True, device_map={"": 0})
processor = AutoProcessor.from_pretrained(quant_path)

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
# Generate output
generation_output = model.generate(
    **inputs, 
    max_new_tokens=512
)

print(processor.decode(generation_output[0], skip_special_tokens=True))