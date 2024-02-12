import os
import gc
import json
import torch
import transformers
import torch.nn as nn

from tqdm import tqdm
from typing import List, Union, Dict
from safetensors.torch import save_file
from typing_extensions import Doc, Annotated
from huggingface_hub import snapshot_download
from transformers.modeling_utils import shard_checkpoint

from awq.modules.linear.gemm import WQLinear_GEMM
from awq.modules.linear.gemv import WQLinear_GEMV
from awq.modules.linear.marlin import WQLinear_Marlin, marlin_post_init
from awq.modules.linear.exllama import WQLinear_Exllama, exllama_post_init
from awq.modules.linear.exllamav2 import WQLinear_ExllamaV2, exllamav2_post_init
from awq.utils.module import (
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    AutoProcessor,
    CLIPImageProcessor,
    PreTrainedTokenizer,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from awq.models._config import AwqConfig
from awq.modules.act import ScaledActivation
from awq.quantize.quantizer import AwqQuantizer
from awq.utils.module import get_named_linears, set_op_by_name

# Since we support different `AutoModelForxxx` from transformers
# we need to define a custom mapping dict as below:
TRANSFORMERS_AUTO_MAPPING_DICT = {
    "mpt": "AutoModelForCausalLM",
    "llama": "AutoModelForCausalLM",
    "opt": "AutoModelForCausalLM",
    "RefinedWeb": "AutoModelForCausalLM",
    "RefinedWebModel": "AutoModelForCausalLM",
    "falcon": "AutoModelForCausalLM",
    "bloom": "AutoModelForCausalLM",
    "gptj": "AutoModelForCausalLM",
    "gpt_bigcode": "AutoModelForCausalLM",
    "mistral": "AutoModelForCausalLM",
    "mixtral": "AutoModelForCausalLM",
    "gpt_neox": "AutoModelForCausalLM",
    "aquila": "AutoModelForCausalLM",
    "Yi": "AutoModelForCausalLM",
    "qwen": "AutoModelForCausalLM",
    "baichuan": "AutoModelForCausalLM",
    "llava": "AutoModelForVision2Seq",
    "qwen2": "AutoModelForCausalLM",
}


class BaseAWQForCausalLM(nn.Module):
    def __init__(
        self,
        model: Annotated[PreTrainedModel, Doc("The pretrained or quantized model.")],
        model_type: Annotated[str, Doc("The model type, found in config.json.")],
        is_quantized: Annotated[
            bool, Doc("Indicates if the current model is quantized.")
        ],
        config: Annotated[PretrainedConfig, Doc("The config of the model.")],
        quant_config: Annotated[
            AwqConfig, Doc("The quantization config of the model.")
        ],
        processor: Annotated[
            AutoProcessor, Doc("An optional processor, e.g. for vision models.")
        ],
    ):
        """The base model for all AutoAWQ models."""
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: AwqConfig = quant_config
        self.processor: CLIPImageProcessor = processor

    def to(self, device: Annotated[str, Doc("The device to move your model to.")]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @torch.no_grad()
    def quantize(
        self,
        tokenizer: Annotated[
            PreTrainedTokenizer, Doc("The tokenizer to use for quantization.")
        ] = None,
        quant_config: Annotated[
            Dict, Doc("The quantization config you want to use.")
        ] = {},
        calib_data: Annotated[
            Union[str, List[str]],
            Doc(
                "The calibration dataset. Either a string pointing to Huggingface or a list of preloaded examples."
            ),
        ] = "pileval",
        split: Annotated[str, Doc("The split of calib_data.")] = "train",
        text_column: Annotated[str, Doc("The text column of calib_data.")] = "text",
        duo_scaling: Annotated[
            bool, Doc("Whether to scale using both w/x or just x.")
        ] = True,
        export_compatible: Annotated[
            bool,
            Doc(
                "This argument avoids real quantization by only applying the scales without quantizing down to FP16."
            ),
        ] = False,
    ):
        """
        The main quantization function that you can use to quantize your model.

        Example:

        ```python
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model_path = "..."
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        model.quantize(tokenizer, quant_config)
        ```
        """
        self.quant_config: AwqConfig = AwqConfig.from_dict(quant_config)

        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert

        self.quantizer = AwqQuantizer(
            self,
            self.model,
            tokenizer,
            self.quant_config.w_bit,
            self.quant_config.q_group_size,
            self.quant_config.zero_point,
            self.quant_config.version,
            calib_data,
            split,
            text_column,
            duo_scaling,
            modules_to_not_convert=self.quant_config.modules_to_not_convert,
            export_compatible=export_compatible,
        )
        self.quantizer.quantize()

        self.is_quantized = True

    @torch.no_grad()
    def pack(self):
        """
        A utility function for the following scenario. Note that save_quantized will
        overwrite existing weights if you use the same quant_path.

        Example:

        ```python
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            export_compatible=True
        )
        model.save_quantized(...)  # produces GGUF/other compat weights
        model.pack(...) # makes the model CUDA compat
        model.save_quantized(...)  # produces CUDA compat weights
        ```
        """
        self.quantizer.pack()

    @staticmethod
    def fuse_layers(model):
        pass

    def save_quantized(
        self,
        save_dir: Annotated[str, Doc("The directory to save your model to.")],
        safetensors: Annotated[
            bool, Doc("Whether to save the model as safetensors or torch files.")
        ] = True,
        shard_size: Annotated[
            str, Doc("The shard size for sharding large models into multiple chunks.")
        ] = "5GB",
    ):
        save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # Save model and config files with empty state dict
        self.model.config.quantization_config = self.quant_config.to_transformers_dict()
        self.model.generation_config.do_sample = True
        self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

        # Vision transformers have a processor
        if self.processor is not None:
            self.processor.save_pretrained(save_dir)

        # Remove empty state dict
        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        # model_name has no extension, add it when saving state_dict
        model_name = "model.safetensors" if safetensors else "pytorch_model.bin"

        # shard checkpoint into chunks (10GB default)
        shards, index = shard_checkpoint(
            self.model.state_dict(), max_shard_size=shard_size, weights_name=model_name
        )

        for shard_file, shard in shards.items():
            if safetensors:
                # safetensors must be in the same memory, so we duplicate and use contiguous memory
                shard = {k: v.clone().contiguous() for k, v in shard.items()}
                save_file(
                    shard, os.path.join(save_dir, shard_file), metadata={"format": "pt"}
                )
            else:
                torch.save(shard, os.path.join(save_dir, shard_file))

        # save shard index
        if index is not None:
            with open(f"{save_dir}/{model_name}.index.json", "w+") as file:
                file.write(json.dumps(index, indent=4))

    @classmethod
    def from_pretrained(
        self,
        model_path: Annotated[str, Doc("A Huggingface path or local path to a model.")],
        model_type: Annotated[str, Doc("The model type, loaded from config.json.")],
        torch_dtype: Annotated[
            torch.dtype,
            Doc(
                "The dtype to load the model as. May not work with other values than float16."
            ),
        ] = torch.float16,
        trust_remote_code: Annotated[
            bool,
            Doc(
                "Useful for Huggingface repositories that have not been integrated into transformers yet."
            ),
        ] = True,
        safetensors: Annotated[
            bool, Doc("Whether to download/load safetensors instead of torch weights.")
        ] = True,
        device_map: Annotated[
            Union[str, Dict],
            Doc(
                "A device map that will be passed onto the model loading method from transformers."
            ),
        ] = None,
        **model_init_kwargs: Annotated[
            Dict,
            Doc(
                "Additional kwargs that are passed to the model during initialization."
            ),
        ],
    ):
        """A method for initialization of pretrained models, usually in FP16."""
        # Get weights path and quant config
        model_weights_path, config, quant_config = self._load_config(
            self, model_path, "", safetensors, trust_remote_code=trust_remote_code
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)

        processor = None
        if target_cls_name == "AutoModelForVision2Seq":
            processor = AutoProcessor.from_pretrained(model_weights_path)
            processor: CLIPImageProcessor = processor.image_processor

        # If not quantized, must load with AutoModelForCausalLM
        model = target_cls.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs,
        )

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=False,
            config=config,
            quant_config=quant_config,
            processor=processor,
        )

    @classmethod
    def from_quantized(
        self,
        model_path: Annotated[str, Doc("A Huggingface path or local path to a model.")],
        model_type: Annotated[str, Doc("The model type, loaded from config.json.")],
        model_filename: Annotated[
            str, Doc("Load a specific model's filename by specifying this argument.")
        ] = "",
        max_seq_len: Annotated[
            int,
            Doc(
                "The maximum sequence cached sequence length of the model. Larger values may increase loading time and memory usage."
            ),
        ] = None,
        torch_dtype: Annotated[
            torch.dtype,
            Doc(
                "The dtype to load the model as. May not work with other values than float16."
            ),
        ] = torch.float16,
        trust_remote_code: Annotated[
            bool,
            Doc(
                "Useful for Huggingface repositories that have not been integrated into transformers yet."
            ),
        ] = True,
        safetensors: Annotated[
            bool, Doc("Whether to download/load safetensors instead of torch weights.")
        ] = True,
        fuse_layers: Annotated[
            bool,
            Doc(
                "Whether to use fused/optimized combination of layers for increased speed."
            ),
        ] = True,
        use_exllama: Annotated[
            bool, Doc("Whether to map the weights to ExLlamaV1 kernels.")
        ] = False,
        use_exllama_v2: Annotated[
            bool, Doc("Whether to map the weights to ExLlamaV2 kernels.")
        ] = False,
        device_map: Annotated[
            Union[str, Dict],
            Doc(
                "A device map that will be passed onto the model loading method from transformers."
            ),
        ] = "balanced",
        offload_folder: Annotated[
            str,
            Doc("The folder ot offload the model to."),
        ] = None,
        **config_kwargs: Annotated[
            Dict,
            Doc(
                "Additional kwargs that are passed to the config during initialization."
            ),
        ],
    ):
        """A method for initialization of a quantized model, usually in INT4."""
        # [STEP 1-2] Load weights path and configs
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            model_filename,
            safetensors,
            trust_remote_code,
            max_seq_len=max_seq_len,
            **config_kwargs,
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)

        # [STEP 3] Load model
        with init_empty_weights():
            model = target_cls.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

        # Prepare WQLinear layers, replace nn.Linear
        self._load_quantized_modules(
            self,
            model,
            quant_config,
            quant_config.version,
            use_exllama=use_exllama,
            use_exllama_v2=use_exllama_v2,
        )

        model.tie_weights()

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_weights_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
        )

        # Dispath to devices
        if fuse_layers:
            self.fuse_layers(model)

        if quant_config.version == "marlin":
            model = marlin_post_init(model)

        elif use_exllama:
            # creates q4 handle
            model = exllama_post_init(model)
        elif use_exllama_v2:
            # creates q4 handle and allocates scratch spaces wrt max_input_len and max_batch_size
            model = exllamav2_post_init(
                model,
                max_input_len=max_seq_len or 2048,
                max_batch_size=int(os.getenv("AWQ_BATCH_SIZE", 1)),
            )

        return self(
            model,
            model_type,
            is_quantized=True,
            config=config,
            quant_config=quant_config,
            processor=None,
        )

    def _load_config(
        self,
        model_path,
        model_filename,
        safetensors=True,
        trust_remote_code=True,
        max_seq_len=4096,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)

        if model_filename != "":
            model_weights_path = model_path + f"/{model_filename}"
        else:
            model_weights_path = model_path

        # [STEP 2] Load config and set sequence length
        # TODO: Create BaseAWQConfig class
        quant_config = AwqConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_seq_len is None and hasattr(self, "max_seq_len_key"):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            # To add the generate support for Multi-modal models as well
            if hasattr(config, "text_config"):
                config.text_config.max_seq_len = getattr(
                    config, self.max_seq_len_key, 2048
                )
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = max_seq_len

        return model_weights_path, config, quant_config

    def _load_quantized_modules(
        self, model, quant_config, version, use_exllama, use_exllama_v2
    ):
        # Real quantization of weights
        assert not (
            version == "gemv" and (use_exllama or use_exllama_v2)
        ), "Exllama kernels only support GEMM version."

        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )

            # Replace activation functions
            self._scale_activations(self, layer)

            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                if version == "marlin":
                    q_linear_module = WQLinear_Marlin
                elif use_exllama:
                    q_linear_module = WQLinear_Exllama
                elif use_exllama_v2:
                    q_linear_module = WQLinear_ExllamaV2
                elif version == "gemm":
                    q_linear_module = WQLinear_GEMM
                elif version == "gemv":
                    q_linear_module = WQLinear_GEMV

                q_linear = q_linear_module.from_linear(
                    module, quant_config.w_bit, quant_config.q_group_size, True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

            torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)

        if scale_dict["is_scalable"]:
            if not isinstance(scale_dict["scale_layer"], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(
                    scale_dict["scale_shape"], dtype=param.dtype, device=param.device
                )

                # scale activation
                scaled_act = ScaledActivation(scale_dict["scale_layer"], scale_like)
                set_op_by_name(layer, scale_dict["scale_name"], scaled_act)
