from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoProcessor
import torch
from PIL import Image
import requests

# quantized
model_path = "llava-hf/llava-v1.6-34b-hf"
quant_path = "./llava-v1.6-34b-hf-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, safetensors=True, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define data loading methods
def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())
# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')

# test
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"

model_path = "./llava-v1.6-34b-hf"
quant_path = "./llava-v1.6-34b-hf-awq"
model = AutoAWQForCausalLM.from_quantized(quant_path,  safetensors=True, device_map="auto")
processor = AutoProcessor.from_pretrained(quant_path)
inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

generation_output = model.generate(**inputs,max_new_tokens=100, repetition_penalty=1.3 ,early_stopping=True,num_beams=5)
awq_out=processor.decode(generation_output[0], skip_special_tokens=True)
print(awq_out)