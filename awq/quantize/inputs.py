import torch
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import List, Union
from awq.utils.calib_data import get_calib_dataset
from transformers import PreTrainedTokenizer, PreTrainedModel

class ActivationStatCollector:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                       calib_data: Union[str, List[str]]="pileval",
                       num_samples=512, seq_len=512, split="train", text_column="text"):
        self.act_scales = {}
        self.hooks = []
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.samples = get_calib_dataset(
            data=calib_data, tokenizer=self.tokenizer, n_samples=num_samples, 
            block_size=seq_len, split=split, text_column=text_column
        )
    
    def forward(self):
        self.model.cuda()

        for sample in tqdm(self.samples, desc="SmoothQuant"):
            self.model(sample.cuda())
        
        self.model.cpu()
    
    def compute_mean_scale_per_layer(self):
        self.mean_scale = {k:torch.mean(v) for k, v in self.act_scales.items()}
        del self.act_scales
        
    def stat_tensor(self, name: str, tensor: torch.Tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()
        if name in self.act_scales:
            self.act_scales[name] = torch.max(self.act_scales[name], comming_max)
        else:
            self.act_scales[name] = comming_max
            
    def stat_input_hook(self, m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        self.stat_tensor(name, x)
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    functools.partial(self.stat_input_hook, name=name)
                )
                self.hooks.append(hook)
            
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []