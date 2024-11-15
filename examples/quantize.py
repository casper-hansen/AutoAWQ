from regex import R
import torch
from awq import AutoAWQForCausalLM
# from awq.models._config import AWQConfig
from transformers import AutoTokenizer
from torchutils.eval import eval_wikitext2
from torchutils.freeze import freeze_seed
freeze_seed()

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
model_path = 'facebook/opt-125m'

model_path = 'meta-llama/Llama-2-7b-chat-hf'
# model_path = "Qwen/Qwen1.5-0.5B-Chat"
model_path = "Qwen/Qwen2.5-7B-Instruct"
"""
perplexity 6.7588
time 3.866  sec
{'perplexity': 6.7588, 'prediction_time': 3.866}
perplexity 7.2708
time 3.567  sec
{'perplexity': 7.2708, 'prediction_time': 3.567}
"""

model_path = 'Qwen/Qwen1.5-0.5B'
"""
perplexity 15.3238
time 2.856  sec
{'perplexity': 15.3238, 'prediction_time': 2.856}


perplexity 14.8191
time 3.566  sec
{'perplexity': 14.8191, 'prediction_time': 3.566}

perplexity 14.8191
time 3.703  sec
{'perplexity': 14.8191, 'prediction_time': 3.703}
"""

quant_path = f"{model_path.replace('/', '__')}-quantized"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

limit = 20
# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# import habana_frameworks.torch.core as htcore
device = torch.device("hpu")

# result = eval_wikitext2(model.to(device), tokenizer, verbose=True, limit=limit)



# # Quantize
# from awq.quantize.quantizer import AwqQuantizer


model.quantize(tokenizer, quant_config=quant_config)
breakpoint()
model = model
delattr(model, "quantizer")
        
from neural_compressor.torch.quantization import prepare, convert, quantize, RTNConfig, get_default_rtn_config

quant_config = RTNConfig(use_sym=False, bits=4, group_size=128)
q_model = quantize(model, quant_config=quant_config)
# model = prepare(model, RTNConfig(use_sym=False, bits=4, group_size=128))
# qmodel = convert(model, RTNConfig())

result = eval_wikitext2(q_model.to(device), tokenizer, verbose=True, limit=limit)
"""
perplexity 16.7339
time 3.566  sec
{'perplexity': 16.7339, 'prediction_time': 3.566}
"""
# # Quantize
# # # Save quantized model
# model.to("cpu")
# model.save_quantized(quant_path)
# tokenizer.save_pretrained(quant_path)

# print(f'Model is quantized and saved at "{quant_path}"')



# model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)
# tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=False, **{"low_cpu_mem_usage": True, "use_cache": False})
# model.cpu()
# result_awq_reload_qmodel = eval_wikitext2(model.to(device), tokenizer, limit=limit)
# print(f"AWQ reloaded model perplexity: {result_awq_reload_qmodel}")