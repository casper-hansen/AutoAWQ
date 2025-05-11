import warnings

warnings.simplefilter("default", DeprecationWarning)

_FINAL_DEV_MESSAGE = """
I have left this message as the final dev message to help you transition.

Important Notice:
- AutoAWQ is officially deprecated and will no longer be maintained.
- The last tested configuration used Torch 2.6.0 and Transformers 4.51.3.
- If future versions of Transformers break AutoAWQ compatibility, please report the issue to the Transformers project.

Alternative:
- AutoAWQ has been adopted by the vLLM Project: https://github.com/vllm-project/llm-compressor

For further inquiries, feel free to reach out:
- X: https://x.com/casper_hansen_
- LinkedIn: https://www.linkedin.com/in/casper-hansen-804005170/
"""

warnings.warn(_FINAL_DEV_MESSAGE, category=DeprecationWarning, stacklevel=1)

__version__ = "0.2.9"
from awq.models.auto import AutoAWQForCausalLM