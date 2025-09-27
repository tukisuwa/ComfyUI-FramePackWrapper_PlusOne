import os
import re
import torch
import safetensors.torch
from tqdm import tqdm
from .diffusers_helper.memory import get_cuda_free_memory_gb

def _load_weights_on_cpu(ckpt_path):
    """Safely loads weights onto the CPU, bypassing ComfyUI's file loader."""
    if ckpt_path.lower().endswith(".safetensors"):
        return safetensors.torch.load_file(ckpt_path, device="cpu")
    else:
        return torch.load(ckpt_path, map_location="cpu")

def parse_module_strengths(strength_string: str) -> list[tuple[re.Pattern, float]]:
    if not strength_string:
        return []
    
    lines = strength_string.strip().split('\n')
    parsed_strengths = []
    for line in lines:
        line = line.strip()
        if not line or '=' not in line:
            continue
        
        parts = line.split('=', 1)
        pattern_str = parts[0].strip()
        try:
            strength_val = float(parts[1].strip())
            pattern = re.compile(pattern_str)
            parsed_strengths.append((pattern, strength_val))
        except (ValueError, re.error) as e:
            print(f"Skipping invalid module strength entry '{line}': {e}")
            
    return parsed_strengths

def merge_lora_to_state_dict(
    model_path:str, lora_infos: list[dict], device: torch.device, gpu_memory_preservation: float = 0.0, lora_vram_strategy: str = "KeepUntilFull"
) -> dict[str, torch.Tensor]:

    processed_loras = []
    for info in lora_infos:
        lora_sd = _load_weights_on_cpu(info["path"])
        keys = list(lora_sd.keys())
        if not keys:
            print(f"LoRA file is empty: {os.path.basename(info['path'])}")
            continue

        if not keys[0].startswith("lora_unet_"):
            is_diffusion_pipe = False
            for key in keys:
                if key.startswith("diffusion_model.") or key.startswith("transformer."):
                    is_diffusion_pipe = True
                    break
            if is_diffusion_pipe:
                 print("Diffusion-pipe (?) LoRA detected, converting...")
                 lora_sd = convert_from_diffusion_pipe_or_something(lora_sd, "lora_unet_")

        is_hunyuan = any("double_blocks" in k or "single_blocks" in k for k in lora_sd.keys())
        if is_hunyuan:
            print("HunyuanVideo LoRA detected, converting to FramePack format")
            lora_sd = convert_hunyuan_to_framepack(lora_sd)

        new_info = info.copy()
        new_info["sd"] = lora_sd
        processed_loras.append(new_info)

    if not processed_loras:
        return _load_weights_on_cpu(model_path)

    return load_safetensors_with_lora(model_path, processed_loras, device, gpu_memory_preservation, lora_vram_strategy)


def convert_from_diffusion_pipe_or_something(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:

    new_weights_sd = {}
    lora_dims = {}
    for key, weight in lora_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            print(f"unexpected key: {key} in diffusers format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]


    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim, dtype=weight.dtype)

    return new_weights_sd


def load_safetensors_with_lora(
    model_path: str,
    lora_infos: list[dict],
    device: torch.device,
    gpu_memory_preservation: float = 0.0,
    lora_vram_strategy: str = "KeepUntilFull"
) -> dict[str, torch.Tensor]:

    state_dict = _load_weights_on_cpu(model_path)
    model_keys = list(state_dict.keys())
    
    print("Calculating final LoRA strengths for each module...")
    for lora_info in lora_infos:
        strength_map = {}
        lora_sd = lora_info.get("sd", {})
        parsed_strengths = parse_module_strengths(lora_info.get("module_strengths", ""))

        for model_weight_key in model_keys:
            if not model_weight_key.endswith(".weight"):
                continue

            lora_name = "lora_unet_" + model_weight_key.rsplit(".", 1)[0].replace(".", "_")
            down_key = lora_name + ".lora_down.weight"

            if down_key in lora_sd:
                final_strength = lora_info["strength"]
                for pattern, strength in parsed_strengths:
                    if pattern.search(model_weight_key):
                        final_strength = strength
                
                if final_strength != 0.0:
                    strength_map[model_weight_key] = final_strength
        
        lora_info["strength_map"] = strength_map

    print(f"Merging {len(lora_infos)} LoRA(s) into state dict...")
    offload_remaining = False
    processed_keys = set()

    for model_weight_key in tqdm(model_keys, desc=f"Merging LoRAs into {os.path.basename(model_path)}"):
        if not model_weight_key.endswith(".weight"):
            continue
        
        processed_keys.add(model_weight_key)

        # Dynamic offloading logic
        if not offload_remaining:
            free_mem_gb = get_cuda_free_memory_gb(device)
            if free_mem_gb <= gpu_memory_preservation:
                if lora_vram_strategy == "Rotate":
                    print(f"VRAM threshold reached ({free_mem_gb:.2f}GB <= {gpu_memory_preservation:.2f}GB). Offloading processed tensors to CPU.")
                    for key in processed_keys:
                        if key in state_dict and state_dict[key].device == device:
                            state_dict[key] = state_dict[key].to('cpu')
                    torch.cuda.empty_cache()
                else: # KeepUntilFull
                    print(f"VRAM threshold reached ({free_mem_gb:.2f}GB <= {gpu_memory_preservation:.2f}GB). Offloading remaining tensors to CPU.")
                    offload_remaining = True
        
        target_device = torch.device('cpu') if offload_remaining else device

        model_weight = state_dict[model_weight_key].to(device, dtype=torch.float32)
        original_dtype = state_dict[model_weight_key].dtype

        for lora_info in lora_infos:
            strength_map = lora_info.get("strength_map", {})
            current_multiplier = strength_map.get(model_weight_key)

            if not current_multiplier:
                continue

            lora_sd = lora_info["sd"]
            lora_name = "lora_unet_" + model_weight_key.rsplit(".", 1)[0].replace(".", "_")
            down_key = lora_name + ".lora_down.weight"
            up_key = lora_name + ".lora_up.weight"
            alpha_key = lora_name + ".alpha"
            
            down_weight = lora_sd[down_key].to(device, dtype=torch.float32)
            up_weight = lora_sd[up_key].to(device, dtype=torch.float32)

            dim = down_weight.size(0)
            alpha = lora_sd.get(alpha_key, torch.tensor(dim)).to(device, dtype=torch.float32)
            scale = alpha / dim if dim != 0 else 1.0

            if len(model_weight.shape) == 2:
                merged_delta = up_weight @ down_weight
            elif down_weight.shape[2:4] == (1, 1):
                merged_delta = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                merged_delta = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)

            model_weight += current_multiplier * merged_delta * scale

        state_dict[model_weight_key] = model_weight.to(target_device, dtype=original_dtype)

    return state_dict


def convert_hunyuan_to_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    new_lora_sd = {}
    for key, weight in lora_sd.items():
        if "double_blocks" in key:
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in key:
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
        else:
            continue

        if "QKVM" in key:
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                if "alpha" not in key and weight.size(1) != 3072:
                    print(f"QKVM weight size mismatch, skipping: {key}. {weight.size()}")
                    continue
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                if weight.size(0) != 21504:
                    print(f"QKVM weight size mismatch, skipping: {key}. {weight.size()}")
                    continue
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :] 
            else:
                print(f"Unsupported module name: {key}")
                continue
        elif "QKV" in key:
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                if "alpha" not in key and weight.size(1) != 3072:
                    print(f"QKV weight size mismatch, skipping: {key}. {weight.size()}")
                    continue
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                if weight.size(0) != 3072 * 3:
                     print(f"QKV weight size mismatch, skipping: {key}. {weight.size()}")
                     continue
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                print(f"Unsupported module name: {key}")
                continue
        else:
            new_lora_sd[key] = weight

    return new_lora_sd