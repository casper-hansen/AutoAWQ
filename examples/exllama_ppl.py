import gc
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from auto_gptq.utils import Perplexity

quant_path = "TheBloke/Llama-2-7B-AWQ"
tokenizer = AutoTokenizer.from_pretrained(quant_path)
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading GEMM model...")
gemm_model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)
gemm_model.device = torch.device("cuda")
gemm_ppl = Perplexity(gemm_model, tokenizer, "wikitext", None, "test", "text")
all_gemm_ppl = gemm_ppl.calculate_perplexity(512, 2048)
print("Mean GEMM PPL:", sum(all_gemm_ppl) / len(all_gemm_ppl))
del gemm_model
gc.collect()
torch.cuda.empty_cache()


print("Loading Exllama model...")
exllama_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama=True
)
exllama_model.device = torch.device("cuda")
exllama_ppl = Perplexity(exllama_model, tokenizer, "wikitext", None, "test", "text")
all_exllama_ppl = exllama_ppl.calculate_perplexity(512, 2048)
print("Mean Exllama PPL:", sum(all_exllama_ppl) / len(all_exllama_ppl))
del exllama_model
gc.collect()
torch.cuda.empty_cache()


print("Loading ExllamaV2 model...")
exllamav2_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama_v2=True
)
exllamav2_model.device = torch.device("cuda")
exllamav2_ppl = Perplexity(exllamav2_model, tokenizer, "wikitext", None, "test", "text")
all_exllamav2_ppl = exllamav2_ppl.calculate_perplexity(512, 2048)
print("Mean ExllamaV2 PPL:", sum(all_exllamav2_ppl) / len(all_exllamav2_ppl))
del exllamav2_model
gc.collect()
torch.cuda.empty_cache()
