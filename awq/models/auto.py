import os
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
}

def check_and_get_model_type(model_dir, trust_remote_code=True, **model_init_kwargs):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code, **model_init_kwargs)
    if config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type

class AutoAWQForCausalLM:
    def __init__(self):
        raise EnvironmentError('You must instantiate AutoAWQForCausalLM with\n'
                               'AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.from_pretrained')
    
    @classmethod
    def from_pretrained(self, model_path, trust_remote_code=True, safetensors=False,
                              device_map=None, **model_init_kwargs) -> BaseAWQForCausalLM:
        model_type = check_and_get_model_type(model_path, trust_remote_code, **model_init_kwargs)

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path, model_type, trust_remote_code=trust_remote_code, safetensors=safetensors,
            device_map=device_map, **model_init_kwargs
        )

    @classmethod
    def from_quantized(self, quant_path, quant_filename='', max_new_tokens=None,
                       trust_remote_code=True, fuse_layers=True,
                       batch_size=1, safetensors=True,
                       device_map="balanced", offload_folder=None, **config_kwargs) -> BaseAWQForCausalLM:
        os.environ["AWQ_BATCH_SIZE"] = str(batch_size)
        model_type = check_and_get_model_type(quant_path, trust_remote_code)

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path, model_type, quant_filename, max_new_tokens, trust_remote_code=trust_remote_code, 
            fuse_layers=fuse_layers, safetensors=safetensors, 
            device_map=device_map, offload_folder=offload_folder,
            **config_kwargs
        )
