import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWQ Quantization entry point')
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help='Path to the pretrained model'
    )

    parser.add_argument(
        '--save_path',
        type = str,
        required = True,
        help = 'Path to save the quantized model. '
    )

    parser.add_argument(
        '--save_to_hf',
        required=False,
        action='store_true',
        help='Save the quantized model to the huggingface hub.'
    )

    parser.add_argument(
        '--repo_name',
        type=str,
        required=False,
        help='Name of the repo to be created in the huggingface hub.'
    )

    parser.add_argument(
        '--repo_is_private',
        required=False,
        action='store_true',
        help='Set this flag to make the repo containing the model private.'
    )

    parser.add_argument(
        '--huggingface_token',
        type=str,
        help='Huggingface token to upload the model to the hub.'
    )

    args = parser.parse_args()

    model = AutoAWQForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Run quantization on the model
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 }
    model.quantize(tokenizer, quant_config=quant_config)

    # Save the quantized model
    model.save_quantized(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    if args.save_to_hf:

        api = HfApi()

        api.create_repo(
            token=args.huggingface_token,
            name=args.repo_name,
            exist_ok=True,
            private=args.repo_is_private,
            repo_type="model",
        )

        api.upload_folder(
            folder_path=args.save_path,
            repo_id=args.repo_name,
            repo_type="model",
            token=args.huggingface_token,
        )
        