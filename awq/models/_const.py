from awq.models import *

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
    "gpt_neox": GPTNeoXAWQForCausalLM,
    "aquila": AquilaAWQForCausalLM,
    "Yi": YiAWQForCausalLM
}

AWQ_FUSER_MAP = {
    "mpt": MptFuser,
    "llama": LlamaFuser,
    "RefinedWeb": FalconFuser,
    "RefinedWebModel": FalconFuser,
    "falcon": FalconFuser,
    "mistral": MistralFuser,
    "aquila": AquilaFuser,
    "Yi": YiFuser
}