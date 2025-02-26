import os
import gc
import warnings
import torch
import transformers
import torch.nn as nn

from tqdm import tqdm
from typing import List, Union, Dict
from typing_extensions import Doc, Annotated
from huggingface_hub import snapshot_download, save_torch_state_dict

from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_IPEX,
    WQLinear_Marlin,
    WQLinear_Exllama,
    WQLinear_ExllamaV2,
    WQLinear_GEMVFast,
    marlin_post_init,
    exllama_post_init,
    exllamav2_post_init,
    ipex_post_init,
)
from awq.utils.module import (
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    try_import,
)
from awq.utils.utils import get_best_device, ipex_available, triton_available
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    AutoProcessor,
    BaseImageProcessor,
    ProcessorMixin,
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
    "exaone": "AutoModelForCausalLM",
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
    "qwen2_vl": "AutoModelForVision2Seq",
    "gemma": "AutoModelForCausalLM",
    "gemma2": "AutoModelForCausalLM",
    "stablelm": "AutoModelForCausalLM",
    "starcoder2": "AutoModelForCausalLM",
    "llava_next": "AutoModelForVision2Seq",
    "phi3": "AutoModelForCausalLM",
    "phi3_v": "AutoModelForCausalLM",
    "cohere": "AutoModelForCausalLM",
    "deepseek_v2": "AutoModelForCausalLM",
    "deepseek_v3": "AutoModelForCausalLM",
    "minicpm": "AutoModelForCausalLM",
    "minicpm3":"AutoModelForCausalLM",
    "internlm2": "AutoModelForCausalLM",
    "qwen2_vl": "AutoModelForVision2Seq",
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
            BaseImageProcessor, Doc("An optional processor, e.g. for vision models.")
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
        self.processor: ProcessorMixin = processor

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
        apply_clip: Annotated[
            bool,
            Doc(
                "Whether to apply clipping to the model during quantization. Some models may perform better with this set to False."
            ),
        ] = True,
        n_parallel_calib_samples: Annotated[
            int,
            Doc(
                "The number of parallel samples to run through the model. "
                "A high number of parallel samples can result in OOM during quantization if max_calib_samples is high enough. "
                "If None, runs through all samples at the same time. "
                "You can set this to a low number for more memory efficient quantization."
            ),
        ] = None,
        max_calib_samples: Annotated[
            int, Doc("The maximum number of samples to run through the model.")
        ] = 128,
        max_calib_seq_len: Annotated[
            int,
            Doc(
                "The maximum sequence length of the calibration dataset. Discard samples greater than max_calib_seq_len."
            ),
        ] = 512,
        max_chunk_memory: Annotated[
            int,
            Doc(
                "The loss computation and per-channel mean is optimized into chunked computations."
                " Adjust this parameter to increase or decrease memory usage for these computations."
                " Default is 1GB (1024 * 1024 * 1024)."
            ),
        ] = 1024
        * 1024
        * 1024,
        quantizer_cls: Annotated[
            AwqQuantizer,
            Doc("If you want to customize the quantization class, you can use AwqQuantizer as a base class.")
        ] = AwqQuantizer,
        twosteps = False,
        shift_scale = False,
        **kwargs,
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

        self.quantizer = quantizer_cls(
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
            apply_clip=apply_clip,
            n_parallel_calib_samples=n_parallel_calib_samples,
            max_calib_samples=max_calib_samples,
            max_calib_seq_len=max_calib_seq_len,
            max_chunk_memory=max_chunk_memory,
            twosteps=twosteps,
            shift_scale=shift_scale,
            **kwargs,
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

        save_torch_state_dict(
            state_dict=self.model.state_dict(),
            save_directory=save_dir,
            max_shard_size=shard_size,
            safe_serialization=safetensors,
            force_contiguous=True,
            shared_tensors_to_discard=self.model._tied_weights_keys,
        )

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
        ] = "auto",
        download_kwargs: Annotated[
            Dict,
            Doc("Used for configure download model"),
        ] = None,
        low_cpu_mem_usage: Annotated[
            bool,
            Doc("Use low_cpu_mem_usage when loading from transformers.")
        ] = True,
        use_cache: Annotated[
            bool,
            Doc("Use use_cache argument in transformers")
        ] = False,
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
            self,
            model_path,
            "",
            safetensors,
            trust_remote_code=trust_remote_code,
            download_kwargs=download_kwargs,
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)

        processor = None
        if target_cls_name == "AutoModelForVision2Seq":
            processor = AutoProcessor.from_pretrained(model_weights_path)

        if model_init_kwargs.get("low_cpu_mem_usage") is None:
            model_init_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if model_init_kwargs.get("use_cache") is None and target_cls_name != "AutoModelForVision2Seq":
            model_init_kwargs["use_cache"] = use_cache

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
        use_ipex: Annotated[
            bool, Doc("Whether to map the weights to ipex kernels for CPU and XPU device.")
        ] = False,
        device_map: Annotated[
            Union[str, Dict],
            Doc(
                "A device map that will be passed onto the model loading method from transformers."
            ),
        ] = "balanced",
        max_memory: Annotated[
            Dict[Union[int, str], Union[int, str]],
            Doc(
                'A dictionary device identifier to maximum memory which will be passed onto the model loading method from transformers. For example：{0: "4GB",1: "10GB"'
            ),
        ] = None,
        offload_folder: Annotated[
            str,
            Doc("The folder ot offload the model to."),
        ] = None,
        download_kwargs: Annotated[
            Dict,
            Doc("Used for configure download model"),
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
            download_kwargs=download_kwargs,
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

        best_device = get_best_device()
        if best_device == "cpu" or (best_device == "xpu:0" and not triton_available):
            use_ipex = True
        if use_ipex and not ipex_available:
            raise ImportError(
                "Please install intel_extension_for_pytorch with "
                "`pip install intel_extension_for_pytorch` for 'ipex' kernel!"
            )
        # Prepare WQLinear layers, replace nn.Linear
        self._load_quantized_modules(
            self,
            model,
            quant_config,
            quant_config.version,
            use_exllama=use_exllama,
            use_exllama_v2=use_exllama_v2,
            use_ipex=use_ipex,
        )

        model.tie_weights()

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_weights_path,
            device_map=device_map,
            max_memory=max_memory,
            no_split_module_classes=[self.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
        )

        # Dispath to devices
        awq_ext, msg = try_import("awq_ext")
        if fuse_layers:
            if best_device in ["mps", "cuda:0"] and awq_ext is None:
                warnings.warn("Skipping fusing modules because AWQ extension is not installed." + msg)
            else:
                self.fuse_layers(model)

        if use_ipex:
            # repack qweight to match the ipex kernel.
            model = ipex_post_init(model)
        elif quant_config.version == "marlin":
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

        model.eval()

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
        download_kwargs=None,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt", "*.onnx*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            if download_kwargs is None:
                download_kwargs = {}

            if "ignore_patterns" in download_kwargs:
                download_kwargs_ignore_patterns = download_kwargs.pop("ignore_patterns")

                if isinstance(download_kwargs_ignore_patterns, str):
                    ignore_patterns.append(download_kwargs_ignore_patterns)
                elif isinstance(download_kwargs_ignore_patterns, list):
                    ignore_patterns.extend(download_kwargs_ignore_patterns)

            model_path = snapshot_download(
                model_path, ignore_patterns=ignore_patterns, **download_kwargs
            )

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
        self, model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
    ):
        # Real quantization of weights
        assert not (
            version == "gemv" and (use_exllama or use_exllama_v2 or use_ipex)
        ), "Exllama kernels only support GEMM version."

        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to include
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )

            # Replace activation functions
            self._scale_activations(self, layer)

            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                if use_ipex:
                    q_linear_module = WQLinear_IPEX
                elif version == "marlin":
                    q_linear_module = WQLinear_Marlin
                elif use_exllama:
                    q_linear_module = WQLinear_Exllama
                elif use_exllama_v2:
                    q_linear_module = WQLinear_ExllamaV2
                elif version == "gemm":
                    q_linear_module = WQLinear_GEMM
                elif version == "gemv":
                    q_linear_module = WQLinear_GEMV
                elif version == "gemv_fast":
                    q_linear_module = WQLinear_GEMVFast


                q_linear = q_linear_module.from_linear(
                    module, quant_config.w_bit, quant_config.q_group_size, True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

            if not use_ipex:
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
