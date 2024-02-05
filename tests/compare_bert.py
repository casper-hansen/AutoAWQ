from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch, time

model_id = "/home/michael/embeddings/bge-small-en-v1.5-quant"

modelawq = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_tks(s="The quick brown fox jumps over the lazy dog"):
    input_text = [s * 100] * 64
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    for k, v in inputs.items():
        inputs[k] = v.to(modelawq.device)
    return inputs

def time_model(m, input_ids, printit=False):
    with torch.inference_mode():
        start = time.time()
        out = m(**input_ids)
        end = time.time()
        if printit:
            print(f"Time: {end - start} {m.config._name_or_path}")
    return out 


warmup = get_tks("This is a long warmup string to warm up the model. Again.")
benchmark = get_tks("This is a long benchmark string to benchmark the model.")
time_model(model, warmup, printit=False)
out_m = time_model(model, benchmark, printit=True)
time_model(modelawq, warmup, printit=False)
out_awq = time_model(modelawq, benchmark, printit=True)

print(out_m.last_hidden_state - out_awq.last_hidden_state)
