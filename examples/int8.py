from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from awq.utils.perplexity_utils import Perplexity

model_path = 'PY007/TinyLlama-1.1B-intermediate-step-480k-1T'
quant_config = { "version": "SmoothQuant", "w_bit": 8 }

model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)

ppl = Perplexity(model.model, tokenizer)
out = ppl.calculate_perplexity()
