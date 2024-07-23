import os
import logging
from transformers import AutoConfig
from awq.models import *
from awq.models.base import BaseAWQForCausalLM


AWQ_CAUSAL_LM_MODEL_MAP = {
    "mpt": MptAWQForCausalLM,
    "llama": LlamaAWQForCausalLM,
    "opt": OptAWQForCausalLM,
    "RefinedWeb": FalconAWQForCausalLM,
    "RefinedWebModel": FalconAWQForCausalLM,
    "falcon": FalconAWQForCausalLM,
    "bloom": BloomAWQForCausalLM,
    "gptj": GPTJAWQForCausalLM,
    "gpt_bigcode": GptBigCodeAWQForCausalLM,
    "mistral": MistralAWQForCausalLM,
    "mixtral": MixtralAWQForCausalLM,
    "gpt_neox": GPTNeoXAWQForCausalLM,
    "aquila": AquilaAWQForCausalLM,
    "Yi": YiAWQForCausalLM,
    "qwen": QwenAWQForCausalLM,
    "baichuan": BaichuanAWQForCausalLM,
    "llava": LlavaAWQForCausalLM,
    "qwen2": Qwen2AWQForCausalLM,
    "gemma": GemmaAWQForCausalLM,
    "stablelm": StableLmAWQForCausalLM,
    "starcoder2": Starcoder2AWQForCausalLM,
    "llava_next": LlavaNextAWQForCausalLM,
    "phi3": Phi3AWQForCausalLM,
    "cohere": CohereAWQForCausalLM,
    "deepseek_v2": DeepseekV2AWQForCausalLM,
    "minicpm": MiniCPMAWQForCausalLM,
}


def check_and_get_model_type(model_dir, trust_remote_code=True, **model_init_kwargs):
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, **model_init_kwargs
    )
    if config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


class AutoAWQForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "You must instantiate AutoAWQForCausalLM with\n"
            "AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.from_pretrained"
        )

    @classmethod
    def from_pretrained(
        self,
        model_path,
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        download_kwargs=None,
        **model_init_kwargs,
    ) -> BaseAWQForCausalLM:
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            model_type,
            trust_remote_code=trust_remote_code,
            safetensors=safetensors,
            device_map=device_map,
            download_kwargs=download_kwargs,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        self,
        quant_path,
        quant_filename="",
        max_seq_len=2048,
        trust_remote_code=True,
        fuse_layers=True,
        use_exllama=False,
        use_exllama_v2=False,
        use_qbits=False,
        batch_size=1,
        safetensors=True,
        device_map="balanced",
        max_memory=None,
        offload_folder=None,
        download_kwargs=None,
        **config_kwargs,
    ) -> BaseAWQForCausalLM:
        os.environ["AWQ_BATCH_SIZE"] = str(batch_size)
        model_type = check_and_get_model_type(quant_path, trust_remote_code)

        if config_kwargs.get("max_new_tokens") is not None:
            max_seq_len = config_kwargs["max_new_tokens"]
            logging.warning(
                "max_new_tokens argument is deprecated... gracefully "
                "setting max_seq_len=max_new_tokens."
            )

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path,
            model_type,
            quant_filename,
            max_seq_len,
            trust_remote_code=trust_remote_code,
            fuse_layers=fuse_layers,
            use_exllama=use_exllama,
            use_exllama_v2=use_exllama_v2,
            use_qbits=use_qbits,
            safetensors=safetensors,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )
