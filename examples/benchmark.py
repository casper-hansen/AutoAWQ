import time
import torch
import argparse
import numpy as np
import pandas as pd
from awq import AutoAWQForCausalLM

def warmup(model):
    warm_up = torch.randn((4096,4096)).to(next(model.parameters()).device)
    torch.mm(warm_up,warm_up)

def generate(model, input_ids, n_generate):
    context_time = 0
    generate_time = 0

    with torch.inference_mode():
        for i in range(n_generate):
            torch.cuda.synchronize()
            start = time.time()

            if i == 0:
                # prefill context
                inputs = torch.as_tensor([input_ids], device=next(model.parameters()).device)
            else:
                # decode tokens
                inputs = torch.as_tensor([[token]], device=next(model.parameters()).device)
            
            out = model(inputs, use_cache=True)

            torch.cuda.synchronize()
            token = out[0][:, -1].max(1)[1].unsqueeze(1)

            if i == 0:
                context_time += time.time() - start
            else:
                generate_time += time.time() - start
    
    return context_time, generate_time

def run_round(model_path, quant_file, n_generate, input_ids):
    print(f" -- Loading model...")
    model = AutoAWQForCausalLM.from_quantized(model_path, quant_file, fuse_layers=True)

    print(f" -- Warming up...")
    warmup(model)

    print(f" -- Generating {n_generate} tokens, {len(input_ids)} token prompt...")
    context_time, generate_time = generate(model, input_ids, n_generate)

    prefill_tokens_per_second = n_generate / context_time
    decode_tokens_per_second = n_generate / generate_time
    memory_used = torch.cuda.max_memory_allocated(next(model.parameters()).device) / (1024 ** 2)

    print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
    print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")
    print(f" ** Max Memory (): {memory_used:.2f} GB")

    return {
        "Prefill tokens/s": prefill_tokens_per_second,
        "Decode tokens/s": decode_tokens_per_second,
        "Memory (VRAM)": memory_used,
        "GPU": torch.cuda.get_device_name()
    }

def main(args):
    rounds = [
        {"context": 4, "n_generate": 200},
        {"context": 32, "n_generate": 32},
        {"context": 64, "n_generate": 64},
        {"context": 128, "n_generate": 128},
        {"context": 256, "n_generate": 256},
        {"context": 512, "n_generate": 512},
        {"context": 1024, "n_generate": 1024},
        {"context": 2048, "n_generate": 2048},
    ]

    all_stats = []

    for settings in rounds:
        input_ids = [1 for _ in range(settings["context"])]

        stats = run_round(
            args.model_path,
            args.quant_file,
            settings["n_generate"],
            input_ids
        )
        
        all_stats.append(stats)
    
    df = pd.DataFrame(all_stats)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="vicuna-7b-v1.5-awq-gemv", help="path to the model")
    parser.add_argument("--quant_file", type=str, default="awq_model_w4_g128.pt", help="weights filename")
    args = parser.parse_args()

    main(args)