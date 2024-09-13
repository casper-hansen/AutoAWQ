from __future__ import annotations

import logging

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor

from awq.models.qwen2vl import Qwen2VLAWQForConditionalGeneration


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load your processor and model with AutoAWQ
processor = Qwen2VLProcessor.from_pretrained(model_path)
model = Qwen2VLAWQForConditionalGeneration.from_pretrained(
    model_path, model_type="qwen2_vl", use_cache=False, attn_implementation="flash_attention_2"
)


# Then you need to prepare your data for calibaration. What you need to do is just put samples into a list,
# each of which is a typical chat message as shown below. you can specify text and image in `content` field:
# dataset = [
#     # message 0
#     [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me who you are."},
#         {"role": "assistant", "content": "I am a large language model named Qwen..."},
#     ],
#     # message 1
#     [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": "file:///path/to/your/image.jpg"},
#                 {"type": "text", "text": "Output all text in the image"},
#             ],
#         },
#         {"role": "assistant", "content": "The text in the image is balabala..."},
#     ],
#     # other messages...
#     ...,
# ]
# here, we use a caption dataset **only for demonstration**. You should replace it with your own sft dataset.
def prepare_dataset(n_sample: int = 8) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]")
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "generate a caption for this image"},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]


dataset = prepare_dataset()

# process the dataset into tensors
text = processor.apply_chat_template(dataset, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(dataset)
inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

# Then just run the calibration process by one line of code:
model.quantize(calib_data=inputs, quant_config=quant_config)

# Finally, save the quantized model:
model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
processor.save_pretrained(quant_path)

# Then you can obtain your own AWQ quantized model for deployment. Enjoy!
