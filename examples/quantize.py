import torch.nn as nn

from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

# Specify paths and hyperparameters for quantization
model_path = "mistral-community/pixtral-12b"
quant_path = "pixtral-12b-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(
    model_path
)
# FIXME: hack to make pixtral work
model.processor.tokenizer.pad_token_id = 11

def print_module_devices(model):
    for name, module in model.named_modules():
        # Check parameters
        param_devices = {
            param_name: param.device 
            for param_name, param in module.named_parameters(recurse=False)
        }
        
        # Check buffers
        buffer_devices = {
            buffer_name: buffer.device 
            for buffer_name, buffer in module.named_buffers(recurse=False)
        }
        
        if param_devices or buffer_devices:
            if param_devices:
                for param_name, device in param_devices.items():
                    print(f" {name} {param_name}: {device}")
            if buffer_devices:
                for buffer_name, device in buffer_devices.items():
                    print(f" {name}  {buffer_name}: {device}")


# We define our own quantizer by extending the AwqQuantizer.
# The main difference is in how the samples are processed when
# the quantization process initialized.
class PixtralAwqQuantizer(AwqQuantizer):
    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.awq_model.get_model_layers(self.model)
        samples = self.calib_data

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)
        
        # FIXME: Hacky way to move the vision part to the right device
        self.model.vision_tower = self.model.vision_tower.to(best_device)
        self.model.multi_modal_projector = self.model.multi_modal_projector.to(best_device)

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

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        print_module_devices(self.model)
        try:
            self.model(**samples.to(best_device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps

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

# Then just run the calibration process by one line of code
model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=PixtralAwqQuantizer)

# Save the model
model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")