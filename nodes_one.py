import os
import torch
import math
from tqdm import tqdm
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
from comfy.cli_args import args, LatentPreviewMethod

from .utils import log

script_directory = os.path.dirname(os.path.abspath(__file__))
vae_scaling_factor = 0.476986

from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModel
from .diffusers_helper.memory import DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .diffusers_helper.bucket_tools import find_nearest_bucket

# Import original function for fallback
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers

def patched_convert_hunyuan_video_lora(original_state_dict):
    """Patched version that handles problematic tensors during conversion"""
    try:
        # Filter out problematic tensors first
        filtered_state_dict = {}
        for key, value in original_state_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    print(f"Skipping 0-dimensional tensor: {key}")
                    continue
                filtered_state_dict[key] = value
            else:
                print(f"Skipping non-tensor value: {key}")
        
        print(f"After filtering: {len(filtered_state_dict)} valid keys")
        
        # First try the original conversion
        try:
            from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
            result = _convert_hunyuan_video_lora_to_diffusers(filtered_state_dict)
            print("Successfully converted LoRA weights")
            return result
        except Exception as e:
            print(f"Error in standard conversion: {e}")
            print("Falling back to custom conversion")
            
            # If standard conversion fails, use the custom implementation
            return converted_state_dict
            
    except Exception as e:
        print(f"LoRA conversion failed: {str(e)}")
        # Return empty state dict as fallback
        return {}
            
    except Exception as e:
        print(f"LoRA conversion failed: {str(e)}")
        # Return empty state dict as fallback
        return {}

        def remap_img_attn_qkv_(key, state_dict):
            try:
                weight = state_dict.pop(key)
                
                # Add dimension check
                if weight.dim() == 0:
                    logger.warning(f"Invalid tensor dimensions for {key}: scalar tensor. Skipping.")
                    return
                    
                if "lora_A" in key:
                    state_dict[key.replace("img_attn_qkv", "attn.to_q")] = weight
                    state_dict[key.replace("img_attn_qkv", "attn.to_k")] = weight
                    state_dict[key.replace("img_attn_qkv", "attn.to_v")] = weight
                else:
                    # Ensure tensor is properly sized before chunking
                    if weight.dim() == 0 or weight.size(0) < 3:
                        logger.warning(f"Invalid tensor size for {key}: {weight.shape}. Using equal splits.")
                        # Create minimal placeholders
                        if weight.dim() > 0 and weight.size(0) > 0:
                            to_q = weight[:1]
                            to_k = weight[:1] if weight.size(0) == 1 else weight[1:2]
                            to_v = weight[:1] if weight.size(0) <= 2 else weight[2:3]
                        else:
                            # For zero-dim tensors, create basic ones
                            to_q = torch.ones(1, dtype=weight.dtype, device=weight.device)
                            to_k = torch.ones(1, dtype=weight.dtype, device=weight.device)
                            to_v = torch.ones(1, dtype=weight.dtype, device=weight.device)
                    else:
                        to_q, to_k, to_v = weight.chunk(3, dim=0)
                        
                    state_dict[key.replace("img_attn_qkv", "attn.to_q")] = to_q
                    state_dict[key.replace("img_attn_qkv", "attn.to_k")] = to_k
                    state_dict[key.replace("img_attn_qkv", "attn.to_v")] = to_v
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}. Skipping.")

        def remap_txt_attn_qkv_(key, state_dict):
            try:
                weight = state_dict.pop(key)
                
                # Add dimension check
                if weight.dim() == 0:
                    logger.warning(f"Invalid tensor dimensions for {key}: scalar tensor. Skipping.")
                    return
                    
                if "lora_A" in key:
                    state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = weight
                    state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = weight
                    state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = weight
                else:
                    # Ensure tensor is properly sized before chunking
                    if weight.dim() == 0 or weight.size(0) < 3:
                        logger.warning(f"Invalid tensor size for {key}: {weight.shape}. Using equal splits.")
                        # Create minimal placeholders
                        if weight.dim() > 0 and weight.size(0) > 0:
                            to_q = weight[:1]
                            to_k = weight[:1] if weight.size(0) == 1 else weight[1:2]
                            to_v = weight[:1] if weight.size(0) <= 2 else weight[2:3]
                        else:
                            # For zero-dim tensors, create basic ones
                            to_q = torch.ones(1, dtype=weight.dtype, device=weight.device)
                            to_k = torch.ones(1, dtype=weight.dtype, device=weight.device)
                            to_v = torch.ones(1, dtype=weight.dtype, device=weight.device)
                    else:
                        to_q, to_k, to_v = weight.chunk(3, dim=0)
                        
                    state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = to_q
                    state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = to_k
                    state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = to_v
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}. Skipping.")

        def remap_txt_in_(key, state_dict):
            def rename_key(key):
                new_key = key.replace("individual_token_refiner.blocks", "token_refiner.refiner_blocks")
                new_key = new_key.replace("adaLN_modulation.1", "norm_out.linear")
                new_key = new_key.replace("txt_in", "context_embedder")
                new_key = new_key.replace("t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1")
                new_key = new_key.replace("t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2")
                new_key = new_key.replace("c_embedder", "time_text_embed.text_embedder")
                new_key = new_key.replace("mlp", "ff")
                return new_key

            try:
                if "self_attn_qkv" in key:
                    weight = state_dict.pop(key)
                    # Ensure tensor is at least 1D before chunking
                    if weight.dim() == 0 or weight.size(0) < 3:
                        logger.warning(f"Invalid tensor dimensions for {key}: {weight.shape}. Skipping.")
                        return
                        
                    to_q, to_k, to_v = weight.chunk(3, dim=0)
                    state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_q"))] = to_q
                    state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_k"))] = to_k
                    state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_v"))] = to_v
                else:
                    state_dict[rename_key(key)] = state_dict.pop(key)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}")
                if key in state_dict:
                    state_dict.pop(key)

        def remap_single_transformer_blocks_(key, state_dict):
            try:
                hidden_size = 3072

                if "linear1.lora_A.weight" in key or "linear1.lora_B.weight" in key:
                    linear1_weight = state_dict.pop(key)
                    if "lora_A" in key:
                        new_key = key.replace("single_blocks", "single_transformer_blocks")
                        if new_key.endswith(".linear1.lora_A.weight"):
                            new_key = new_key[:-len(".linear1.lora_A.weight")]
                        state_dict[f"{new_key}.attn.to_q.lora_A.weight"] = linear1_weight
                        state_dict[f"{new_key}.attn.to_k.lora_A.weight"] = linear1_weight
                        state_dict[f"{new_key}.attn.to_v.lora_A.weight"] = linear1_weight
                        state_dict[f"{new_key}.proj_mlp.lora_A.weight"] = linear1_weight
                    else:
                        # Ensure tensor size is sufficient for splitting
                        if linear1_weight.dim() == 0 or linear1_weight.size(0) < 3 * hidden_size:
                            logger.warning(f"Invalid tensor size for {key}: {linear1_weight.shape}. Skipping splitting.")
                            return
                            
                        split_size = (hidden_size, hidden_size, hidden_size, linear1_weight.size(0) - 3 * hidden_size)
                        q, k, v, mlp = torch.split(linear1_weight, split_size, dim=0)
                        new_key = key.replace("single_blocks", "single_transformer_blocks")
                        if new_key.endswith(".linear1.lora_B.weight"):
                            new_key = new_key[:-len(".linear1.lora_B.weight")]
                        state_dict[f"{new_key}.attn.to_q.lora_B.weight"] = q
                        state_dict[f"{new_key}.attn.to_k.lora_B.weight"] = k
                        state_dict[f"{new_key}.attn.to_v.lora_B.weight"] = v
                        state_dict[f"{new_key}.proj_mlp.lora_B.weight"] = mlp

                elif "linear1.lora_A.bias" in key or "linear1.lora_B.bias" in key:
                    linear1_bias = state_dict.pop(key)
                    if "lora_A" in key:
                        new_key = key.replace("single_blocks", "single_transformer_blocks")
                        if new_key.endswith(".linear1.lora_A.bias"):
                            new_key = new_key[:-len(".linear1.lora_A.bias")]
                        state_dict[f"{new_key}.attn.to_q.lora_A.bias"] = linear1_bias
                        state_dict[f"{new_key}.attn.to_k.lora_A.bias"] = linear1_bias
                        state_dict[f"{new_key}.attn.to_v.lora_A.bias"] = linear1_bias
                        state_dict[f"{new_key}.proj_mlp.lora_A.bias"] = linear1_bias
                    else:
                        # Ensure tensor size is sufficient for splitting
                        if linear1_bias.dim() == 0 or linear1_bias.size(0) < 3 * hidden_size:
                            logger.warning(f"Invalid tensor size for {key}: {linear1_bias.shape}. Skipping splitting.")
                            return
                            
                        split_size = (hidden_size, hidden_size, hidden_size, linear1_bias.size(0) - 3 * hidden_size)
                        q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, split_size, dim=0)
                        new_key = key.replace("single_blocks", "single_transformer_blocks")
                        if new_key.endswith(".linear1.lora_B.bias"):
                            new_key = new_key[:-len(".linear1.lora_B.bias")]
                        state_dict[f"{new_key}.attn.to_q.lora_B.bias"] = q_bias
                        state_dict[f"{new_key}.attn.to_k.lora_B.bias"] = k_bias
                        state_dict[f"{new_key}.attn.to_v.lora_B.bias"] = v_bias
                        state_dict[f"{new_key}.proj_mlp.lora_B.bias"] = mlp_bias

                else:
                    new_key = key.replace("single_blocks", "single_transformer_blocks")
                    new_key = new_key.replace("linear2", "proj_out")
                    new_key = new_key.replace("q_norm", "attn.norm_q")
                    new_key = new_key.replace("k_norm", "attn.norm_k")
                    state_dict[new_key] = state_dict.pop(key)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}")
                if key in state_dict:
                    state_dict.pop(key)

        TRANSFORMER_KEYS_RENAME_DICT = {
            "img_in": "x_embedder",
            "time_in.mlp.0": "time_text_embed.timestep_embedder.linear_1",
            "time_in.mlp.2": "time_text_embed.timestep_embedder.linear_2",
            "guidance_in.mlp.0": "time_text_embed.guidance_embedder.linear_1",
            "guidance_in.mlp.2": "time_text_embed.guidance_embedder.linear_2",
            "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
            "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
            "double_blocks": "transformer_blocks",
            "img_attn_q_norm": "attn.norm_q",
            "img_attn_k_norm": "attn.norm_k",
            "img_attn_proj": "attn.to_out.0",
            "txt_attn_q_norm": "attn.norm_added_q",
            "txt_attn_k_norm": "attn.norm_added_k",
            "txt_attn_proj": "attn.to_add_out",
            "img_mod.linear": "norm1.linear",
            "img_norm1": "norm1.norm",
            "img_norm2": "norm2",
            "img_mlp": "ff",
            "txt_mod.linear": "norm1_context.linear",
            "txt_norm1": "norm1.norm",
            "txt_norm2": "norm2_context",
            "txt_mlp": "ff_context",
            "self_attn_proj": "attn.to_out.0",
            "modulation.linear": "norm.linear",
            "pre_norm": "norm.norm",
            "final_layer.norm_final": "norm_out.norm",
            "final_layer.linear": "proj_out",
            "fc1": "net.0.proj",
            "fc2": "net.2",
            "input_embedder": "proj_in",
        }

        TRANSFORMER_SPECIAL_KEYS_REMAP = {
            "txt_in": remap_txt_in_,
            "img_attn_qkv": remap_img_attn_qkv_,
            "txt_attn_qkv": remap_txt_attn_qkv_,
            "single_blocks": remap_single_transformer_blocks_,
            "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
        }

        # Some folks attempt to make their state dict compatible with diffusers by adding "transformer." prefix to all keys
        # and use their custom code. To make sure both "original" and "attempted diffusers" loras work as expected, we make
        # sure that both follow the same initial format by stripping off the "transformer." prefix.
        for key in list(converted_state_dict.keys()):
            try:
                if key.startswith("transformer."):
                    converted_state_dict[key[len("transformer.") :]] = converted_state_dict.pop(key)
                if key.startswith("diffusion_model."):
                    converted_state_dict[key[len("diffusion_model.") :]] = converted_state_dict.pop(key)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}")

        # Rename and remap the state dict keys
        for key in list(converted_state_dict.keys()):
            try:
                new_key = key[:]
                for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
                    new_key = new_key.replace(replace_key, rename_key)
                converted_state_dict[new_key] = converted_state_dict.pop(key)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}")

        for key in list(converted_state_dict.keys()):
            try:
                for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
                    if special_key not in key:
                        continue
                    handler_fn_inplace(key, converted_state_dict)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}")

        # Add back the "transformer." prefix
        for key in list(converted_state_dict.keys()):
            try:
                converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {str(e)}")

        return converted_state_dict
    except Exception as e:
        logger.error(f"LoRA conversion failed: {str(e)}")
        # Return empty state dict as fallback
        return {}


class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = mm.get_torch_device()

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

class FramePackTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable single block compilation"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable double block compilation"}),
            },
        }
    RETURN_TYPES = ("FRAMEPACKCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_single_blocks": compile_single_blocks,
            "compile_double_blocks": compile_double_blocks
        }

        return (compile_args, )

class DownloadAndLoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["lllyasviel/FramePackI2V_HY"],),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa"):

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        device = mm.get_torch_device()

        model_path = os.path.join(folder_paths.models_dir, "diffusers", "lllyasviel", "FramePackI2V_HY")
        if not os.path.exists(model_path):
            print(f"Downloading clip model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_path, torch_dtype=base_dtype, attention_mode=attention_mode).cpu()
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == 'fp8_e4m3fn' or quantization == 'fp8_e4m3fn_fast':
            transformer = transformer.to(torch.float8_e4m3fn)
            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
        elif quantization == 'fp8_e5m2':
            transformer = transformer.to(torch.float8_e5m2)
        else:
            transformer = transformer.to(base_dtype)

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

class FramePackLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"),
                {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "fuse_lora": ("BOOLEAN", {"default": True, "tooltip": "Fuse the LORA model with the base model. This is recommended for better performance."}),
            },
            "optional": {
                "prev_lora":("FPLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
            }
        }

    RETURN_TYPES = ("FPLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, prev_lora=None, fuse_lora=True):
        loras_list = []

        lora = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "fuse_lora": fuse_lora,
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)

class LoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "cuda", "tooltip": "Initialize the model on the main device or offload device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
                "lora": ("FPLORA", {"default": None, "tooltip": "LORA model to load"}),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa", lora=None, load_device="main_device"):
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if load_device == "main_device":
            transformer_load_device = device
        else:
            transformer_load_device = offload_device
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        model_config_path = os.path.join(script_directory, "transformer_config.json")
        import json
        with open(model_config_path, "r") as f:
            config = json.load(f)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        model_weight_dtype = sd['single_transformer_blocks.0.attn.to_k.weight'].dtype
        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModel(**config, attention_mode=attention_mode)
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype
        if lora is not None:
            after_lora_dtype = dtype
            dtype = base_dtype
        print("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(),
                desc=f"Loading transformer parameters to {transformer_load_device}",
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
            set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])

        if lora is not None:
            adapter_list = []
            adapter_weights = []

            for l in lora:
                fuse = True if l["fuse_lora"] else False
                lora_sd = load_torch_file(l["path"])

                if "lora_unet_single_transformer_blocks_0_attn_to_k.lora_up.weight" in lora_sd:
                    from .utils import convert_to_diffusers
                    lora_sd = convert_to_diffusers("lora_unet_", lora_sd)

                if not "transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight" in lora_sd:
                    log.info(f"Converting LoRA weights from {l['path']} to diffusers format...")
                    state_dict_copy = {}
                    
                    # Remove scalar (0-dimensional) tensors
                    for key, value in lora_sd.items():
                        if isinstance(value, torch.Tensor):
                            if value.dim() == 0:
                                print(f"Skipping 0-dimensional tensor: {key}")
                                continue
                            state_dict_copy[key] = value
                        else:
                            print(f"Skipping non-tensor value: {key}")
                    
                    print(f"After filtering: {len(state_dict_copy)} valid keys")
                    
                    try:
                        from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
                        lora_sd = _convert_hunyuan_video_lora_to_diffusers(state_dict_copy)
                        print("Successfully converted LoRA weights")
                    except Exception as e:
                        print(f"Error in standard conversion: {e}")
                        # Fall back to empty dict if conversion fails
                        print("Conversion failed, returning empty state dict")
                        lora_sd = {}

                lora_rank = None
                for key, val in lora_sd.items():
                    if "lora_B" in key or "lora_up" in key:
                        lora_rank = val.shape[1]
                        break
                if lora_rank is not None:
                    log.info(f"Merging rank {lora_rank} LoRA weights from {l['path']} with strength {l['strength']}")
                    adapter_name = l['path'].split("/")[-1].split(".")[0]
                    adapter_weight = l['strength']
                    transformer.load_lora_adapter(lora_sd, weight_name=l['path'].split("/")[-1], lora_rank=lora_rank, adapter_name=adapter_name)

                    adapter_list.append(adapter_name)
                    adapter_weights.append(adapter_weight)

                del lora_sd
                mm.soft_empty_cache()
            if adapter_list:
                transformer.set_adapters(adapter_list, weights=adapter_weights)
                if fuse:
                    if model_weight_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        raise ValueError("Fusing LoRA doesn't work well with fp8 model weights. Please use a bf16 model file, or disable LoRA fusing.")
                    lora_scale = 1
                    transformer.fuse_lora(lora_scale=lora_scale)
                    transformer.delete_adapters(adapter_list)

            if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_e5m2":
                params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
                for name, param in transformer.named_parameters():
                    # Do not cast LoRA weights to fp8
                    if not any(keyword in name for keyword in params_to_keep) and not 'lora' in name:
                        param.data = param.data.to(after_lora_dtype)

        if quantization == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear
            convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

class FramePackFindNearestBucket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "base_resolution": ("INT", {"default": 640, "min": 64, "max": 2048, "step": 16, "tooltip": "Width of the image to encode"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("width","height",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Finds the closes resolution bucket as defined in the orignal code"

    def process(self, image, base_resolution):

        H, W = image.shape[1], image.shape[2]

        new_height, new_width = find_nearest_bucket(H, W, resolution=base_resolution)

        return (new_width, new_height, )

# For video generation
class FramePackSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "start_latent": ("LATENT", {"tooltip": "init Latents to use for image2video"} ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The size of the latent window to use for sampling."}),
                "total_second_length": ("FLOAT", {"default": 5, "min": 1, "max": 120, "step": 0.1, "tooltip": "The total length of the video in seconds."}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"],
                    {
                        "default": 'unipc_bh1'
                    }),
            },
            "optional": {
                "image_embeds": ("CLIP_VISION_OUTPUT", ),
                "end_latent": ("LATENT", {"tooltip": "end Latents to use for image2video"} ),
                "end_image_embeds": ("CLIP_VISION_OUTPUT", {"tooltip": "end Image's clip embeds"} ),
                "embed_interpolation": (["disabled", "weighted_average", "linear"], {"default": 'disabled', "tooltip": "Image embedding interpolation type. If linear, will smoothly interpolate with time, else it'll be weighted average with the specified weight."}),
                "start_embed_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weighted average constant for image embed interpolation. If end image is not set, the embed's strength won't be affected"}),
                "initial_samples": ("LATENT", {"tooltip": "init Latents to use for video2video"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"

    def process(self, model, shift, positive, negative, latent_window_size, use_teacache, total_second_length, teacache_rel_l1_thresh, steps, cfg,
                guidance_scale, seed, sampler, gpu_memory_preservation, start_latent=None, image_embeds=None, end_latent=None, end_image_embeds=None, embed_interpolation="linear", start_embed_strength=1.0, initial_samples=None, denoise_strength=1.0):
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print("total_latent_sections: ", total_latent_sections)

        transformer = model["transformer"]
        base_dtype = model["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if start_latent is not None:
            start_latent = start_latent["samples"] * vae_scaling_factor
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
        if end_latent is not None:
            end_latent = end_latent["samples"] * vae_scaling_factor
        has_end_image = end_latent is not None
        print("start_latent", start_latent.shape)
        B, C, T, H, W = start_latent.shape

        if image_embeds is not None:
            start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(device, base_dtype)

        if has_end_image:
            assert end_image_embeds is not None
            end_image_encoder_last_hidden_state = end_image_embeds["last_hidden_state"].to(device, base_dtype)
        else:
            if image_embeds is not None:
                end_image_encoder_last_hidden_state = torch.zeros_like(start_image_encoder_last_hidden_state)

        llama_vec = positive[0][0].to(device, base_dtype)
        clip_l_pooler = positive[0][1]["pooled_output"].to(device, base_dtype)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(device, base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(device, base_dtype)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Sampling
        rnd = torch.Generator("cpu").manual_seed(seed)

        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, H, W), dtype=torch.float32).cpu()

        total_generated_latent_frames = 0

        latent_paddings_list = list(reversed(range(total_latent_sections)))
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        comfy_model = HyVideoModel(
                HyVideoModelConfig(base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=device,
            )

        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

        if total_latent_sections > 4:
            # Duplicating some padding items seems to produce better results when total_latent_sections > 4
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            latent_paddings_list = latent_paddings.copy()

        for i, latent_padding in enumerate(latent_paddings):
            print(f"latent_padding: {latent_padding}")
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            if image_embeds is not None:
                if embed_interpolation != "disabled":
                    if embed_interpolation == "linear":
                        if total_latent_sections <= 1:
                            frac = 1.0
                        else:
                            frac = 1 - i / (total_latent_sections - 1)
                    else:
                        frac = start_embed_strength if has_end_image else 1.0

                    image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * frac + (1 - frac) * end_image_encoder_last_hidden_state
                else:
                    image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * start_embed_strength
            else:
                image_encoder_last_hidden_state = None

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            start_latent_frames = T
            indices = torch.arange(0, sum([start_latent_frames, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([start_latent_frames, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Use end image latent for the first section if provided
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # vid2vid
            if initial_samples is not None:
                total_length = initial_samples.shape[2]
                max_padding = max(latent_paddings_list)

                if is_last_section:
                    # Last section captures the end of the sequence
                    start_idx = max(0, total_length - latent_window_size)
                else:
                    # Distribute windows evenly across the sequence
                    if max_padding > 0:
                        progress = (max_padding - latent_padding) / max_padding
                        start_idx = int(progress * max(0, total_length - latent_window_size))
                    else:
                        start_idx = 0

                end_idx = min(start_idx + latent_window_size, total_length)
                print(f"start_idx: {start_idx}, end_idx: {end_idx}, total_length: {total_length}")
                input_init_latents = initial_samples[:, :, start_idx:end_idx, :, :].to(device)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True):
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler=sampler,
                    initial_latent=input_init_latents if initial_samples is not None else None,
                    strength=denoise_strength,
                    width=W * 8,
                    height=H * 8,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=guidance_scale,
                    guidance_rescale=0,
                    shift=shift if shift != 0 else None,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device,
                    dtype=base_dtype,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if is_last_section:
                break

        transformer.to(offload_device)
        mm.soft_empty_cache()

        return {"samples": real_history_latents / vae_scaling_factor},

class FramePackSingleFrameSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "start_latent": ("LATENT", {"tooltip": "init Latents to use for image2image"} ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The size of the latent window to use for sampling."}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"], {"default": 'unipc_bh1'}),
                "use_kisekaeichi": ("BOOLEAN", {"default": False, "tooltip": "Enable Kisekaeichi mode for style transfer"}),
            },
            "optional": {
                "image_embeds": ("CLIP_VISION_OUTPUT", ),
                "initial_samples": ("LATENT", {"tooltip": "init Latents to use for image2image variation"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_latent": ("LATENT", {"tooltip": "Reference image latent for kisekaeichi mode"}),
                "reference_image_embeds": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference image CLIP embeds for kisekaeichi mode"}),
                "target_index": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1, "tooltip": "Target index for kisekaeichi mode (recommended: 1)"}),
                "history_index": ("INT", {"default": 13, "min": 0, "max": 16, "step": 1, "tooltip": "History index for kisekaeichi mode (recommended: 13)"}),
                "input_mask": ("MASK", {"tooltip": "Input mask for selective application"}),
                "reference_mask": ("MASK", {"tooltip": "Reference mask for selective features"}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Single frame sampler with Kisekaeichi (style transfer) support"

    def process(self, model, shift, positive, negative, latent_window_size, use_teacache, teacache_rel_l1_thresh, steps, cfg,
                guidance_scale, seed, sampler, gpu_memory_preservation, start_latent=None, image_embeds=None,
                initial_samples=None, denoise_strength=1.0, use_kisekaeichi=False, reference_latent=None,
                reference_image_embeds=None, target_index=1, history_index=13, input_mask=None, reference_mask=None,
                # MagCache Parameters are optional and passed from the experimental node
                enable_magcache=False, magcache_mag_ratios_str="", magcache_retention_ratio=0.2,
                magcache_threshold=0.24, magcache_k=6, magcache_calibration=False,
                # RoPE Parameters are optional and passed from the experimental node
                rope_scaling_factor=1.0, rope_scaling_timestep_threshold=1001,
                # Advanced One-Frame Mode parameters
                advanced_one_frame_mode=False, one_frame_mode_str=""):

        print("=== Single Frame Inference Mode ===")
        transformer = model["transformer"]
        base_dtype = model["dtype"]

        if hasattr(transformer, 'enable_teacache'):
            transformer.enable_teacache = False
        if hasattr(transformer, 'enable_magcache'):
            transformer.enable_magcache = False

        if hasattr(transformer, 'rope_scaling_factor'):
            transformer.rope_scaling_factor = rope_scaling_factor
        if hasattr(transformer, 'rope_scaling_timestep_threshold'):
            transformer.rope_scaling_timestep_threshold = rope_scaling_timestep_threshold if rope_scaling_timestep_threshold <= 1000 else None

        if enable_magcache:
            mag_ratios = None
            if magcache_mag_ratios_str and not magcache_calibration:
                try:
                    mag_ratios = [float(x.strip()) for x in magcache_mag_ratios_str.split(',') if x.strip()]
                except ValueError:
                    print(f"Warning: Could not parse magcache_mag_ratios_str: {magcache_mag_ratios_str}. Using default ratios.")
                    mag_ratios = None

            print(f"Initializing MagCache with enable={enable_magcache}, retention_ratio={magcache_retention_ratio}, threshold={magcache_threshold}, K={magcache_k}, calibration={magcache_calibration}")
            if hasattr(transformer, 'initialize_magcache'):
                transformer.initialize_magcache(
                    enable=enable_magcache,
                    retention_ratio=magcache_retention_ratio,
                    mag_ratios=mag_ratios,
                    magcache_thresh=magcache_threshold,
                    K=magcache_k,
                    calibration=magcache_calibration,
                )
            else:
                print("Warning: Transformer model does not have 'initialize_magcache' method. MagCache will not be used.")
                enable_magcache = False

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if start_latent is not None:
            start_latent = start_latent["samples"] * vae_scaling_factor
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
        if use_kisekaeichi and reference_latent is not None:
            reference_latent = reference_latent["samples"] * vae_scaling_factor
            print(f"Reference image latent: {reference_latent.shape}")
        
        print("start_latent", start_latent.shape)
        B, C, T, H, W = start_latent.shape

        if image_embeds is not None:
            start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(device, base_dtype)
        else:
            start_image_encoder_last_hidden_state = None

        if use_kisekaeichi and reference_image_embeds is not None:
            reference_image_encoder_last_hidden_state = reference_image_embeds["last_hidden_state"].to(device, base_dtype)
            print("Reference image CLIP embedding set")
        else:
            reference_image_encoder_last_hidden_state = None

        llama_vec = positive[0][0].to(device, base_dtype)
        clip_l_pooler = positive[0][1]["pooled_output"].to(device, base_dtype)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(device, base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(device, base_dtype)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        rnd = torch.Generator("cpu").manual_seed(seed)
        sample_num_frames = 1
        
        image_encoder_last_hidden_state = start_image_encoder_last_hidden_state
        
        # Basic single-frame generation setup
        clean_latents = start_latent
        clean_latent_indices = torch.tensor([[0]], dtype=torch.long)
        latent_indices = torch.tensor([[latent_window_size - 1]], dtype=torch.long)
        clean_latents_2x = None
        clean_latent_2x_indices = None
        clean_latents_4x = None
        clean_latent_4x_indices = None

        # Kisekaeichi mode is a preset for advanced mode
        if use_kisekaeichi and not advanced_one_frame_mode:
            print("--- Kisekaeichi Mode (Standard) ---")
            advanced_one_frame_mode = True
            one_frame_mode_str = f"target_index={target_index},control_indices=0;{history_index},no_2x,no_4x,no_post"

        if advanced_one_frame_mode:
            print(f"--- Advanced One-Frame Mode ---")
            print(f"Mode string: '{one_frame_mode_str}'")
            commands = {s.strip() for s in one_frame_mode_str.split(',')}
            
            control_latents_list = []
            
            masked_start_latent = start_latent
            if input_mask is not None:
                mask_resized = common_upscale(input_mask.unsqueeze(0).unsqueeze(0), W, H, "bilinear", "center").squeeze(0).squeeze(0)
                masked_start_latent = start_latent * mask_resized.to(start_latent.device)[None, None, None, :, :]
            control_latents_list.append(masked_start_latent)

            if reference_latent is not None:
                masked_ref_latent = reference_latent
                if reference_mask is not None:
                    mask_resized = common_upscale(reference_mask.unsqueeze(0).unsqueeze(0), W, H, "bilinear", "center").squeeze(0).squeeze(0)
                    masked_ref_latent = reference_latent * mask_resized.to(reference_latent.device)[None, None, None, :, :]
                control_latents_list.append(masked_ref_latent)

            clean_latents = torch.cat(control_latents_list, dim=2)

            latent_indices = torch.tensor([[latent_window_size - 1]], dtype=torch.long)
            clean_latent_indices = torch.arange(clean_latents.shape[2], dtype=torch.long).unsqueeze(0)

            for cmd in commands:
                if cmd.startswith("target_index="):
                    latent_indices[0, 0] = int(cmd.split('=')[1])
                elif cmd.startswith("control_indices="):
                    indices_str = cmd.split('=')[1].split(';')
                    clean_latent_indices = torch.tensor([int(i) for i in indices_str], dtype=torch.long).unsqueeze(0)
            
            if "no_2x" not in commands:
                clean_latents_2x = torch.zeros((B, C, 2, H, W), dtype=start_latent.dtype)
                clean_latent_2x_indices = torch.arange(latent_window_size + 1, latent_window_size + 3).unsqueeze(0)
            if "no_4x" not in commands:
                clean_latents_4x = torch.zeros((B, C, 16, H, W), dtype=start_latent.dtype)
                clean_latent_4x_indices = torch.arange(latent_window_size + 3, latent_window_size + 19).unsqueeze(0)

            if "no_post" in commands:
                pass
            
            if use_kisekaeichi and reference_image_embeds is not None and start_image_encoder_last_hidden_state is not None:
                ref_weight = 0.3
                image_encoder_last_hidden_state = torch.lerp(start_image_encoder_last_hidden_state, reference_image_embeds["last_hidden_state"].to(device, base_dtype), ref_weight)

            print(f"  - Final clean_latents shape: {clean_latents.shape}")
            print(f"  - Final latent_indices: {latent_indices}")
            print(f"  - Final clean_latent_indices: {clean_latent_indices}")
            print(f"  - Clean 2x enabled: {clean_latents_2x is not None}")
            print(f"  - Clean 4x enabled: {clean_latents_4x is not None}")
            
        input_init_latents = None
        if initial_samples is not None:
            input_init_latents = initial_samples[:, :, 0:1, :, :].to(device)
            print("Initial samples set")

        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

        if use_teacache:
            if enable_magcache:
                print("Warning: TEACache and MagCache are both enabled. Disabling TEACache.")
                transformer.initialize_teacache(enable_teacache=False)
            else:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_rel_l1_thresh)
        else:
            transformer.initialize_teacache(enable_teacache=False)

        if enable_magcache and hasattr(transformer, 'reset_magcache'):
            transformer.reset_magcache(steps)

        print("=== Starting Sampling ===")
        print(f"sample_num_frames: {sample_num_frames}")
        print(f"clean_latents frames used: {clean_latents.shape[2] if clean_latents is not None else 0}")
        print(f"clean_latent_2x_indices: {clean_latent_2x_indices}")
        print(f"clean_latent_4x_indices: {clean_latent_4x_indices}")
        
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True):
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler=sampler,
                initial_latent=input_init_latents,
                strength=denoise_strength,
                width=W * 8,
                height=H * 8,
                frames=sample_num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=guidance_scale,
                guidance_rescale=0,
                shift=shift if shift != 0 else None,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=device,
                dtype=base_dtype,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

        if enable_magcache and magcache_calibration:
            try:
                if hasattr(transformer, 'get_calibration_data'):
                    norm_ratio, norm_std, cos_dis = transformer.get_calibration_data()
                    print("\nMagCache Calibration Data:")
                    print(f"  - norm_ratio: {norm_ratio}")
                    print(f"  - norm_std: {norm_std}")
                    print(f"  - cos_dis: {cos_dis}")
                    print("\nSuggested --magcache_mag_ratios (copy and paste):")
                    suggested_ratios = [1.0] + norm_ratio
                    print(",".join([f"{ratio:.5f}" for ratio in suggested_ratios]))
                else:
                    print("Warning: Transformer model does not have 'get_calibration_data' method.")
            except Exception as e:
                print(f"Error getting MagCache calibration data: {e}")

        if not getattr(self, 'keep_model_in_vram', False):
            transformer.to(offload_device)
            mm.soft_empty_cache()
            print("Model offloaded from VRAM.")
        else:
            print("Model kept in VRAM.")

        mode_info = "Advanced One-Frame" if advanced_one_frame_mode else "Kisekaeichi" if use_kisekaeichi else "Normal"
        print(f"=== Single frame generation complete ({mode_info} mode): {generated_latents.shape} ===")
        
        return {"samples": generated_latents / vae_scaling_factor},


class FramePackSingleFrameSamplerExperimental:
    @classmethod
    def INPUT_TYPES(s):
        types = FramePackSingleFrameSampler.INPUT_TYPES()
        types["required"]["keep_model_in_vram"] = ("BOOLEAN", {"default": False, "tooltip": "Keep model in VRAM after generation for faster subsequent runs. May cause OOM with other nodes."})
        
        magcache_inputs = {
            "enable_magcache": ("BOOLEAN", {"default": False, "tooltip": "Enable MagCache for faster inference."}),
            "magcache_mag_ratios_str": ("STRING", {"default": "", "multiline": False, "tooltip": "Comma-separated float values for MagCache ratios. Leave empty for default."}),
            "magcache_retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "magcache_threshold": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 1.0, "step": 0.01}),
            "magcache_k": ("INT", {"default": 6, "min": 1, "max": 100}),
            "magcache_calibration": ("BOOLEAN", {"default": False, "tooltip": "Enable MagCache calibration mode."}),
        }
        types["required"].update(magcache_inputs)

        rope_inputs = {
            "rope_scaling_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01, "tooltip": "RoPE scaling factor for H/W dimensions. Default is 1.0 (no scaling)."}),
            "rope_scaling_timestep_threshold": ("INT", {"default": 1001, "min": 0, "max": 1001, "step": 1, "tooltip": "Timestep threshold to start applying RoPE scaling. 1001 to disable."}),
        }
        types["required"].update(rope_inputs)

        one_frame_inputs = {
            "advanced_one_frame_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable advanced one-frame inference mode to override Kisekaeichi settings."}),
            "one_frame_target_index": ("INT", {"default": 1, "min": 0, "max": 32, "step": 1, "tooltip": "Target index for generation."}),
            "one_frame_control_index_1": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1, "tooltip": "First control index (e.g., for start_latent)."}),
            "one_frame_control_index_2": ("INT", {"default": 13, "min": 0, "max": 32, "step": 1, "tooltip": "Second control index (e.g., for reference_latent)."}),
            "one_frame_no_2x": ("BOOLEAN", {"default": True, "tooltip": "Disable 2x clean latents."}),
            "one_frame_no_4x": ("BOOLEAN", {"default": True, "tooltip": "Disable 4x clean latents."}),
            "one_frame_no_post": ("BOOLEAN", {"default": True, "tooltip": "Disable post clean latents."}),
        }
        types["optional"].update(one_frame_inputs)
        
        if "one_frame_mode_str" in types["optional"]:
            del types["optional"]["one_frame_mode_str"]

        return types

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Single frame sampler with Kisekaeichi (style transfer) support and experimental features"

    def process(self, model, shift, positive, negative, latent_window_size, use_teacache, teacache_rel_l1_thresh, steps, cfg,
                guidance_scale, seed, sampler, gpu_memory_preservation, keep_model_in_vram,
                enable_magcache, magcache_mag_ratios_str, magcache_retention_ratio, magcache_threshold, magcache_k, magcache_calibration,
                rope_scaling_factor, rope_scaling_timestep_threshold,
                start_latent=None, image_embeds=None, 
                initial_samples=None, denoise_strength=1.0, use_kisekaeichi=False, reference_latent=None, 
                reference_image_embeds=None, target_index=1, history_index=13, input_mask=None, reference_mask=None,
                advanced_one_frame_mode=False, 
                one_frame_target_index=1, one_frame_control_index_1=0, one_frame_control_index_2=13,
                one_frame_no_2x=True, one_frame_no_4x=True, one_frame_no_post=True):
        
        original_sampler = FramePackSingleFrameSampler()
        original_sampler.keep_model_in_vram = keep_model_in_vram

        # Construct the one_frame_mode_str from UI inputs
        one_frame_mode_str = ""
        if advanced_one_frame_mode:
            commands = []
            commands.append(f"target_index={one_frame_target_index}")
            commands.append(f"control_indices={one_frame_control_index_1};{one_frame_control_index_2}")
            if one_frame_no_2x:
                commands.append("no_2x")
            if one_frame_no_4x:
                commands.append("no_4x")
            if one_frame_no_post:
                commands.append("no_post")
            one_frame_mode_str = ", ".join(commands)

        result = original_sampler.process(
            model, shift, positive, negative, latent_window_size, use_teacache, teacache_rel_l1_thresh, steps, cfg,
            guidance_scale, seed, sampler, gpu_memory_preservation, start_latent, image_embeds, 
            initial_samples, denoise_strength, use_kisekaeichi, reference_latent, 
            reference_image_embeds, target_index, history_index, input_mask, reference_mask,
            enable_magcache=enable_magcache, magcache_mag_ratios_str=magcache_mag_ratios_str, 
            magcache_retention_ratio=magcache_retention_ratio, magcache_threshold=magcache_threshold, 
            magcache_k=magcache_k, magcache_calibration=magcache_calibration,
            rope_scaling_factor=rope_scaling_factor, rope_scaling_timestep_threshold=rope_scaling_timestep_threshold,
            advanced_one_frame_mode=advanced_one_frame_mode, one_frame_mode_str=one_frame_mode_str
        )
        
        return result


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFramePackModel": DownloadAndLoadFramePackModel,
    "FramePackSampler": FramePackSampler,
    "FramePackSingleFrameSampler": FramePackSingleFrameSampler,
    "FramePackSingleFrameSamplerExperimental": FramePackSingleFrameSamplerExperimental,
    "FramePackTorchCompileSettings": FramePackTorchCompileSettings,
    "FramePackFindNearestBucket": FramePackFindNearestBucket,
    "LoadFramePackModel": LoadFramePackModel,
    "FramePackLoraSelect": FramePackLoraSelect,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFramePackModel": "(Down)Load FramePackModel",
    "FramePackSampler": "FramePackSampler",
    "FramePackSingleFrameSampler": "FramePack Single Frame Sampler",
    "FramePackSingleFrameSamplerExperimental": "FramePack Single Frame Sampler (Experimental)",
    "FramePackTorchCompileSettings": "Torch Compile Settings",
    "FramePackFindNearestBucket": "Find Nearest Bucket",
    "LoadFramePackModel": "Load FramePackModel",
    "FramePackLoraSelect": "Select Lora",
    }
