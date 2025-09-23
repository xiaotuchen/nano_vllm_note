import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    '''in qwen3.py:
    class Qwen3ForCausalLM(nn.Module):
    	packed_modules_mapping = {
    		"q_proj": ("qkv_proj", "q"), # where found key q_proj, its corresponding values are ("qkv_proj", "q")
    		"k_proi": ("qkv_proj", "k"),
    		"v_proj": ("qkv_proj", "v"),
    		"gate_proj": ("gate_up_proj", 0),
    		"up_proj": ("gate_up_proj", 1),
    	}
    '''
    
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    # print(f"{weight_name}, {f.get_tensor(weight_name).shape}") #to print all param with its shape
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k] # ie. when k=q_proj, its v=qkv_proj shard_id=q
                        param_name = weight_name.replace(k, v) # ie. inside weight_name, when found q_proj, replace it with qkv_proj
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
