import os
import re
import torch
from comfy.utils import load_torch_file
from tqdm import tqdm

def parse_module_strengths(strength_string: str) -> list[tuple[re.Pattern, float]]:
    """Parses the module strength string into a list of (regex, strength) tuples."""
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
    model_path:str, lora_infos: list[dict], device: torch.device,
    max_vram_gb: float = 4.0, prewarm_vram_gb: float = 2.0, offload_strategy: str = "Batch Offload"
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.
    """
    processed_loras = []
    for info in lora_infos:
        lora_sd = load_torch_file(info["path"])
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
        return load_torch_file(model_path)

    # Extract settings from the first LoRA info, assuming they are consistent for a given load operation.
    if processed_loras:
        max_vram_gb_actual = processed_loras[0].get("max_vram_gb", max_vram_gb)
        prewarm_vram_gb_actual = processed_loras[0].get("prewarm_vram_gb", prewarm_vram_gb)
        offload_strategy_actual = processed_loras[0].get("offload_strategy", offload_strategy)
    else:
        max_vram_gb_actual = max_vram_gb
        prewarm_vram_gb_actual = prewarm_vram_gb
        offload_strategy_actual = offload_strategy

    return load_safetensors_with_lora(model_path, processed_loras, device, max_vram_gb_actual, prewarm_vram_gb_actual, offload_strategy_actual)


def convert_from_diffusion_pipe_or_something(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to the format used by the diffusion pipeline to Musubi Tuner.
    Copy from Musubi Tuner repo.
    """
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


from safetensors import safe_open

def load_safetensors_with_lora(
    model_path: str,
    lora_infos: list[dict],
    device: torch.device,
    max_vram_gb: float,
    prewarm_vram_gb: float,
    offload_strategy: str,
) -> dict[str, torch.Tensor]:
    """
    Merges LoRA weights into a model's state dict with memory efficiency.
    Supports multiple offloading strategies.
    """
    final_state_dict = {}
    offload_device = torch.device("cpu")
    max_vram_bytes = max_vram_gb * (1024**3)
    prewarm_vram_bytes = prewarm_vram_gb * (1024**3)

    with safe_open(model_path, framework="pt", device="cpu") as f:
        model_keys_sorted = list(f.keys())

    # 1. Pre-calculate strength maps for all LoRAs
    all_lora_modules = set()
    for lora_info in lora_infos:
        lora_sd = lora_info.get("sd", {})
        parsed_strengths = parse_module_strengths(lora_info.get("module_strengths", ""))
        strength_map = {}
        for key in model_keys_sorted:
            if not key.endswith(".weight"): continue
            lora_unet_name = "lora_unet_" + key.rsplit(".", 1)[0].replace(".", "_")
            down_key = lora_unet_name + ".lora_down.weight"
            if down_key not in lora_sd: continue
            
            final_strength = lora_info["strength"]
            for pattern, strength in parsed_strengths:
                if pattern.search(key): final_strength = strength
            if final_strength != 0.0:
                strength_map[key] = final_strength
                all_lora_modules.add(key)
        lora_info["strength_map"] = strength_map

    print(f"Merging {len(all_lora_modules)} modules from {len(lora_infos)} LoRA(s) and loading model...")
    pbar = tqdm(total=len(model_keys_sorted), desc=f"Loading & Merging ({offload_strategy})")

    with safe_open(model_path, framework="pt", device="cpu") as f:
        # Strategy 1: Batch Offload
        if offload_strategy == "Batch Offload":
            tensors_in_vram = {}
            current_vram_usage = 0

            def offload_batch():
                nonlocal current_vram_usage
                if not tensors_in_vram: return
                pbar.set_description(f"Offloading {len(tensors_in_vram)} tensors")
                for k, v_tensor in tensors_in_vram.items():
                    final_state_dict[k] = v_tensor.to(offload_device, non_blocking=True)
                tensors_in_vram.clear()
                if device.type == 'cuda': torch.cuda.empty_cache()
                current_vram_usage = 0
            
            for key in model_keys_sorted:
                base_tensor = f.get_tensor(key)
                # Merge on VRAM
                if key in all_lora_modules:
                    merged_tensor = base_tensor.clone().to(device, dtype=torch.float32)
                    for lora_info in lora_infos:
                        strength_map = lora_info.get("strength_map", {})
                        if key not in strength_map: continue
                        lora_sd, lora_unet_name = lora_info["sd"], "lora_unet_" + key.rsplit(".", 1)[0].replace(".", "_")
                        down_key, up_key, alpha_key = f"{lora_unet_name}.lora_down.weight", f"{lora_unet_name}.lora_up.weight", f"{lora_unet_name}.alpha"
                        try:
                            down_w, up_w = lora_sd[down_key].to(device, dtype=torch.float32), lora_sd[up_key].to(device, dtype=torch.float32)
                            dim, alpha = down_w.shape[0], lora_sd.get(alpha_key, torch.tensor(down_w.shape[0])).to(device, dtype=torch.float32)
                            scale, multiplier = (alpha / dim) if dim != 0 else 1.0, strength_map[key]
                            delta = (up_w @ down_w) if merged_tensor.ndim != 4 else torch.nn.functional.conv2d(down_w.permute(1,0,2,3), up_w).permute(1,0,2,3)
                            merged_tensor += delta * scale * multiplier
                            del down_w, up_w, delta
                        except Exception as e: print(f"Error merging key {key} for LoRA {lora_info['name']}: {e}")
                    final_tensor = merged_tensor.to(dtype=base_tensor.dtype)
                else:
                    final_tensor = base_tensor.to(device)

                if (current_vram_usage + final_tensor.nbytes) > max_vram_bytes:
                    offload_batch()
                
                tensors_in_vram[key] = final_tensor
                current_vram_usage += final_tensor.nbytes
                del base_tensor
                pbar.update(1)

            for k, v_tensor in tensors_in_vram.items():
                final_state_dict[k] = v_tensor
            tensors_in_vram.clear()

        # Strategy 2: Simple Pre-warm (Merge on CPU)
        else: 
            for key in model_keys_sorted:
                base_tensor = f.get_tensor(key)
                if key in all_lora_modules:
                    merged_tensor = base_tensor.clone().to(torch.float32)
                    for lora_info in lora_infos:
                        strength_map = lora_info.get("strength_map", {})
                        if key not in strength_map: continue
                        lora_sd, lora_unet_name = lora_info["sd"], "lora_unet_" + key.rsplit(".", 1)[0].replace(".", "_")
                        down_key, up_key, alpha_key = f"{lora_unet_name}.lora_down.weight", f"{lora_unet_name}.lora_up.weight", f"{lora_unet_name}.alpha"
                        try:
                            down_w, up_w = lora_sd[down_key].to(torch.float32), lora_sd[up_key].to(torch.float32)
                            dim, alpha = down_w.shape[0], lora_sd.get(alpha_key, torch.tensor(down_w.shape[0])).to(torch.float32)
                            scale, multiplier = (alpha / dim) if dim != 0 else 1.0, strength_map[key]
                            delta = (up_w @ down_w) if merged_tensor.ndim != 4 else torch.nn.functional.conv2d(down_w.permute(1,0,2,3), up_w).permute(1,0,2,3)
                            merged_tensor += delta * scale * multiplier
                        except Exception as e: print(f"Error merging key {key} for LoRA {lora_info['name']}: {e}")
                    final_state_dict[key] = merged_tensor.to(dtype=base_tensor.dtype)
                else:
                    final_state_dict[key] = base_tensor
                pbar.update(1)

    pbar.close()

    # 3. Pre-warm specified amount of VRAM (Common for both strategies)
    print(f"Pre-warming up to {prewarm_vram_gb} GB of VRAM...")
    vram_loaded_bytes = 0
    for key in model_keys_sorted:
        if vram_loaded_bytes >= prewarm_vram_bytes:
            break
        tensor = final_state_dict[key]
        if tensor.device == offload_device:
            final_state_dict[key] = tensor.to(device)
            vram_loaded_bytes += tensor.nbytes

    if device.type == 'cuda': torch.cuda.empty_cache()
    print("Model loading and merging complete.")
    return final_state_dict


def convert_hunyuan_to_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert HunyuanVideo LoRA weights to FramePack format.
    """
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        if "double_blocks" in key:
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in key:
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")  # split later
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
        else:
            # This is not necessarily an error, could be text encoder LoRA
            # print(f"Unsupported module name: {key}, only double_blocks and single_blocks are supported")
            continue

        if "QKVM" in key:
            # split QKVM into Q, K, V, M
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                # copy QKVM weight or alpha to Q, K, V, M
                if "alpha" not in key and weight.size(1) != 3072:
                    print(f"QKVM weight size mismatch, skipping: {key}. {weight.size()}")
                    continue
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                # split QKVM weight into Q, K, V, M
                if weight.size(0) != 21504:
                    print(f"QKVM weight size mismatch, skipping: {key}. {weight.size()}")
                    continue
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]  # 21504 - 3072 * 3 = 12288
            else:
                print(f"Unsupported module name: {key}")
                continue
        elif "QKV" in key:
            # split QKV into Q, K, V
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                # copy QKV weight or alpha to Q, K, V
                if "alpha" not in key and weight.size(1) != 3072:
                    print(f"QKV weight size mismatch, skipping: {key}. {weight.size()}")
                    continue
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                # split QKV weight into Q, K, V
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
            # no split needed
            new_lora_sd[key] = weight

    return new_lora_sd
