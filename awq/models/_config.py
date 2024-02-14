import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from transformers.utils.hub import PushToHubMixin, cached_file


@dataclass
class AwqConfig(PushToHubMixin):
    quant_method: str = field(default="awq")
    zero_point: bool = field(default=True)
    q_group_size: int = field(default=128)
    w_bit: int = field(default=4)
    version: str = field(default="gemm")
    config_file_name = "config.json"
    modules_to_not_convert: Optional[List] = None

    @classmethod
    def from_dict(cls, quant_config: Dict = {}):
        if not quant_config:
            quant_config = cls()
        else:
            quant_config = cls(**quant_config)
            quant_config.version = quant_config.version.lower()

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
        else:  # Remote
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

        quant_config = None
        if os.path.exists(resolved_config_file):
            with open(resolved_config_file, "r", encoding="utf-8") as file:
                loaded_config = json.loads(file.read())

            quant_config = loaded_config.get("quantization_config")

            if quant_config is not None:
                awq_config = cls.from_transformers_dict(cls, quant_config)
                quant_config = cls(**awq_config)

        if quant_config is None:
            quant_config = cls()

        return quant_config

    def to_dict(self):
        return {
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "version": self.version,
            "modules_to_not_convert": self.modules_to_not_convert,
        }

    def to_transformers_dict(self):
        return {
            "quant_method": self.quant_method,
            "zero_point": self.zero_point,
            "group_size": self.q_group_size,
            "bits": self.w_bit,
            "version": self.version.lower(),
            "modules_to_not_convert": self.modules_to_not_convert,
        }

    def from_transformers_dict(self, transformers_dict: Dict):
        return {
            "quant_method": transformers_dict.get("quant_method"),
            "zero_point": transformers_dict.get("zero_point"),
            "q_group_size": transformers_dict.get("group_size"),
            "w_bit": transformers_dict.get("bits"),
            "version": transformers_dict.get("version"),
            "modules_to_not_convert": transformers_dict.get("modules_to_not_convert"),
        }
