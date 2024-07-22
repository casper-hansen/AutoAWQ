import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="CLI for model quantization and saving")
    parser.add_argument("--hf_model_path", type=str, required=True, help="Path to the Hugging Face model")
    parser.add_argument("--quant_name", type=str, required=True, help="Name of the quantized model")
    parser.add_argument("--local_save_path", type=str, required=True, help="Path to save the quantized model")

    # Quantization config arguments
    parser.add_argument("--zero_point", action="store_true", help="Enable zero point for quantization")
    parser.add_argument("--no-zero_point", action="store_false", dest="zero_point", help="Disable zero point for quantization")
    parser.add_argument("--q_group_size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--w_bit", type=int, default=4, help="Weight bit width")
    parser.add_argument("--version", type=str, default="GEMM", help="Quantization version")

    # Model config arguments
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="Use low CPU memory")
    parser.add_argument("--no-low_cpu_mem_usage", action="store_false", dest="low_cpu_mem_usage", help="Don't use low CPU memory")
    parser.add_argument("--use_cache", action="store_true", help="Use cache")
    parser.add_argument("--no-use_cache", action="store_false", dest="use_cache", help="Don't use cache")

    parser.set_defaults(zero_point=True, low_cpu_mem_usage=True, use_cache=False)

    args = parser.parse_args()

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }

    model_config = {
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "use_cache": args.use_cache
    }

    print(f"Loading model from: {args.hf_model_path}")
    model = AutoAWQForCausalLM.from_pretrained(args.hf_model_path, **model_config)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)

    print(f"Quantizing model with config: {quant_config}")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving quantized model to: {args.local_save_path}")
    model.save_quantized(args.local_save_path)
    tokenizer.save_pretrained(args.local_save_path)

    print(f"Quantized model '{args.quant_name}' saved successfully.")

if __name__ == "__main__":
    main()