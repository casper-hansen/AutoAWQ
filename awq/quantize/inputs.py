import torch
import functools
import torch.nn as nn
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel

class ActivationStatCollector:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                       dataset, num_samples=512, seq_len=512):
        self.act_scales = {}
        self.hooks = []
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_samples = num_samples
        self.seq_len = seq_len
    
    def forward(self):
        for i in range(self.num_samples):
            device = next(self.model.parameters()).device

            input_ids = self.tokenizer(
                self.dataset[i]["text"], return_tensors="pt",
                max_length=self.seq_len, truncation=True
            ).input_ids.to(device)

            self.model(input_ids)
        
    def stat_tensor(self, name: str, tensor: torch.Tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in self.act_scales:
            self.act_scales[name] = torch.max(self.act_scales[name], comming_max)
        else:
            self.act_scales[name] = comming_max
            
    def stat_input_hook(self, m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        self.stat_tensor(name, x)
    
    def register_hooks(self, layers: List[nn.Linear]):
        for linear in layers:
            hook = linear.register_forward_hook(
                functools.partial(self.stat_input_hook, name=linear._get_name())
            )
            self.hooks.append(hook)
            
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
