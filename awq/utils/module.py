import inspect
import torch.nn as nn
from typing import Dict, Any


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
    if modules_to_not_convert is None:
        return linear_layers

    filtered_layers = {}
    for name, linear_layer in linear_layers.items():
        if not any(key in name for key in modules_to_not_convert):
            filtered_layers[name] = linear_layer
    return filtered_layers


def recreate_module(original_module: nn.Module) -> nn.Module:
    """
    Recreate a PyTorch module with the same structure and parameters as the original.
    
    Args:
        original_module (nn.Module): The original module to recreate.
    
    Returns:
        nn.Module: A new instance of the same type as the original module.
    """
    # Get the class of the original module
    module_class = type(original_module)

    # Get the __init__ parameters of the class
    init_signature = inspect.signature(module_class.__init__)
    init_params = init_signature.parameters

    # Prepare arguments for the new instance
    init_args: Dict[str, Any] = {}

    for name, param in init_params.items():
        if name == 'self':
            continue
        if hasattr(original_module, name):
            init_args[name] = getattr(original_module, name)
        elif param.default != inspect.Parameter.empty:
            init_args[name] = param.default
        else:
            raise ValueError(f"Cannot determine value for parameter '{name}' in {module_class.__name__}")

    # Create a new instance
    new_module = module_class(**init_args)

    # Copy the state dict to ensure all parameters and buffers are the same
    new_module.load_state_dict(original_module.state_dict())

    return new_module
