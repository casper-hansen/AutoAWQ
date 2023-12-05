from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, GenerationConfig
import torch
import glob
import os.path
from datasets import load_dataset
import json
from tqdm import tqdm

model_path = '/root/workspace/external_data/model/base'
quant_path = '/root/workspace/external_data/model/quant'
data_path = '/root/workspace/external_data/data'
predict_output_path = "/workspace/AutoAWQ/predict_result"

# Load model
model = AutoAWQForCausalLM.from_quantized(
    quant_path,
    fuse_layers=True,
)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

# Convert prompt to tokens
prompt_template = 'You are PULSE, a large language model of Transformer architecture trained by OpenMedLab. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-06-28'

answer_start_ids = [tokenizer.convert_tokens_to_ids("<|aim_start|>")]
end_token = '<|im_end|>'
eos_token_ids = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
]
suppress_token_ids = [
    tokenizer.eos_token_id,
]
max_retry = 5
for test_file_path in sorted(
        glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)):
    predict_file_path = test_file_path.replace(data_path, predict_output_path)
    print(f"run eval on {test_file_path}")
    print(f"save eval on {predict_file_path}")

    if os.path.exists(predict_file_path) == True:
        print(f"{predict_file_path} is finish, continue")
        continue

    test_dataset = load_dataset(
        "json",
        data_files=test_file_path,
        split="train",
    )
    predict_output = []
    for data in tqdm(test_dataset):
        retry = 0
        question = data['question']
        input_ids = tokenizer(
            f"<|iim_start|>{prompt_template}<|im_end|>").input_ids
        input_ids += tokenizer(f"<|uim_start|>{question}<|im_end|>",
                               add_special_tokens=False).input_ids
        input_ids += answer_start_ids
        start_pos = len(input_ids)
        tokens = torch.tensor([input_ids]).cuda()
        generation_config = GenerationConfig(
            max_length=16384,
            max_new_tokens=2048,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
            suppress_tokens=suppress_token_ids)
        try:
            for layer in model.model.model.layers:
                layer.self_attn.start_pos = 0
        except Exception:
            for layer in model.model.layers:
                layer.self_attn.start_pos = 0

        while retry < max_retry:
            try:
                generation_output = model.generate(
                    tokens,
                    generation_config,
                )
                break
            except Exception as e:
                print(e)
                retry += 1
                print('retry')

        predict_output += generation_output[:, start_pos:]

    os.makedirs(os.path.dirname(predict_file_path), exist_ok=True)

    with open(predict_file_path, "w", encoding="utf8") as f:
        for test_dataset_item, predict_output_item in zip(
                test_dataset, predict_output):
            f.write(
                json.dumps(
                    {
                        "type":
                        test_dataset_item["type"],
                        "question":
                        test_dataset_item["question"],
                        "reference_answer":
                        test_dataset_item["reference_answer"],
                        "predict_answer":
                        tokenizer.decode(predict_output_item).strip().split(
                            end_token)[0],
                    },
                    ensure_ascii=False) + "\n")
