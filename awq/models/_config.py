import os
import json
from typing import Dict
from dataclasses import dataclass, field
from transformers.utils.hub import PushToHubMixin, cached_file

@dataclass
class AwqConfig(PushToHubMixin):
    quant_method: str = field(default="awq")
    zero_point: bool = field(default=True)
    q_group_size: int = field(default=128)
    w_bit: int = field(default=4)
    version: str = field(default="GEMM")
    config_file_name = "quant_config.json"

    def save_pretrained(self, save_dir: str, **kwargs):
        # quant_config.json
        quant_config = self.to_dict()
        with open(os.path.join(save_dir, self.config_file_name), "w+", encoding="utf-8") as file:
            file.write(json.dumps(quant_config, indent=4))
        
        # config.json: quantization_config
        config_filepath = os.path.join(save_dir, "config.json")
        with open(config_filepath, 'r', encoding="utf-8") as file:
            model_config = json.loads(file.read())
        
        model_config["quantization_config"] = self.to_transformers_dict()

        with open(config_filepath, "w+", encoding="utf-8") as file:
            file.write(json.dumps(model_config, indent=4))
    
    @classmethod
    def from_dict(cls, quant_config: Dict={}):
        if not quant_config:
            quant_config = cls.to_dict(cls)

        if "version" not in quant_config.keys():
            quant_config["version"] = cls.version
        
        return quant_config

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)

        if os.path.isdir(save_dir):  # Local
            resolved_config_file = os.path.join(save_dir, cls.config_file_name)
        else: # Remote
            resolved_config_file = cached_file(
                save_dir,
                cls.config_file_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                revision=revision,
                local_files_only=local_files_only,
                subfolder=subfolder,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _commit_hash=commit_hash,
            )
        
        with open(resolved_config_file, 'r', encoding="utf-8") as file:
            quant_config = json.loads(file.read())

        if "version" not in quant_config.keys():
            quant_config["version"] = cls.version
        
        return quant_config

    def to_dict(self):
        return {
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "version": self.version
        }

    def to_transformers_dict(self):
        return {
            "quant_method": self.quant_method,
            "zero_point": self.zero_point,
            "group_size": self.q_group_size,
            "bits": self.w_bit,
            "version": self.version.lower(),
        }
