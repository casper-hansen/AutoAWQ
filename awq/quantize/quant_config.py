import json
import os
from dataclasses import dataclass, field, fields
from transformers.utils.hub import PushToHubMixin, cached_file
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class QuantConfig(PushToHubMixin):
    """Deeply inspired from https://github.com/PanQiWei/AutoGPTQ/blob/main/auto_gptq/modeling/_base.py
    """

    zero_point: bool = field(default=True)
    q_group_size: int = field(default=128)
    w_bit: int = field(default=4)
    version: str = field(default="GEMM")
    config_file_name = "quant_config.json"

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(os.path.join(save_dir, self.config_file_name), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        """Load a configuration from file.

        :param save_dir: Directory from where to load the file.
        :type save_dir: str
        :return: Initialized QuantConfig
        :rtype: QuantConfig
        """
        
        # Parameters related to loading from Hugging Face Hub
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
        
        field_names = [field.name for field in fields(cls)]
        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)
            filtered_args = {}
            for key, val in args_from_json.items():
                if key in field_names:
                    filtered_args[key] = val
                else:
                    logger.warning(f"ignoring unknown parameter in {cls.config_file_name}: {key}.")
            return cls(**filtered_args)

    def to_dict(self):
        return {
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "version": self.version
        }
