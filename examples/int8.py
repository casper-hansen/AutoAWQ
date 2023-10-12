from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from awq.utils.perplexity_utils import Perplexity

model_path = 'JackFram/llama-68m'
quant_config = { "version": "SmoothQuant", "w_bit": 8 }

model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)

tokens = tokenizer(
    "Hello",
    return_tensors='pt'
).input_ids.cuda()

generation_output = model.generate(
    tokens,
    max_new_tokens=512
)

print(tokenizer.decode(generation_output[0]))

# ppl = Perplexity(model.model, tokenizer)
# out = ppl.calculate_perplexity()
