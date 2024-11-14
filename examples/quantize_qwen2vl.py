import torch
import torch.nn as nn

from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

# Specify paths and hyperparameters for quantization
model_path = "Qwen/Qwen2-VL-7B-Instruct"
quant_path = "qwen2-vl-7b-instruct"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(
    model_path, use_cache=False, attn_implementation="flash_attention_2"
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
text = model.processor.apply_chat_template(dataset, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(dataset)
inputs = model.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

class Qwen2VLAwqQuantizer(AwqQuantizer):
    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.awq_model.get_model_layers(self.model)
        samples = self.calib_data

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
            def get_device(obj: torch.Tensor | nn.Module):
                if isinstance(obj, torch.Tensor):
                    return obj.device
                return next(obj.parameters()).device

            if get_device(obj) != device:
                obj = obj.to(device)
            return obj

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        for k, v in samples.items():
            if isinstance(v, (torch.Tensor, nn.Module)):
                samples[k] = move_to_device(v, best_device)
        try:
            self.model(**samples)
        except ValueError:  # work with early exit
            pass
        finally:
            for k, v in samples.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    samples[k] = move_to_device(v, "cpu")
        modules[0] = modules[0].module  # restore

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps

# Then just run the calibration process by one line of code
model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=Qwen2VLAwqQuantizer)

# Save the model
model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")