import torch

def perform_operation_on_low_mem_device(tensor, operation, low_mem_device, **operation_kwargs):
    """
    AWQ sequentially offloads layers when they have been processed.
    - This function helps execute the next operation on the lowest memory GPU device.
    - Finds the GPU with lowest memory usage, places tensor and operation on device,
      then runs the operation, and places the tensor and operation on the original device.
    """
    original_device = tensor.device
    torch.cuda.synchronize(original_device)
    
    # Move tensor to low memory device
    tensor = tensor.to(low_mem_device)

    for param in operation.parameters():
        param.to(low_mem_device)
    
    # Perform operation
    result = operation(tensor, **operation_kwargs)
    
    # Move tensor back to original device
    if isinstance(result, tuple):
        result = result[0].to(original_device)
    else:
        result = result.to(original_device)

    for param in operation.parameters():
        param.to(original_device)
    
    return result


def get_lowest_memory_device():
    device_id = torch.cuda.current_device()
    min_memory = torch.cuda.memory_allocated(device_id)

    for i in range(torch.cuda.device_count()):
        used_memory = torch.cuda.memory_allocated(i)

        if used_memory < min_memory:
            min_memory = used_memory
            device_id = i

    return torch.device(device_id)
