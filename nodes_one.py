import logging
import math

import torch

logger = logging.getLogger(__name__)

import comfy.latent_formats
import comfy.model_base
import comfy.model_management as mm
from comfy.utils import common_upscale

from .diffusers_helper.memory import move_model_to_device_with_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .nodes import HyVideoModel, HyVideoModelConfig

vae_scaling_factor = 0.476986


class FramePackSingleFrameSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "start_latent": (
                    "LATENT",
                    {"tooltip": "init Latents to use for image2image"},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "tooltip": "Sampling steps (10=fast preview, 20=balanced, 30=high quality)"}),
                "use_teacache": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Use TeaCache for faster sampling (recommended for 15+ steps)."},
                ),
                "teacache_rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The threshold for the relative L1 loss.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01},
                ),
                "shift": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "latent_window_size": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 33,
                        "step": 1,
                        "tooltip": "The size of the latent window to use for sampling.",
                    },
                ),
                "gpu_memory_preservation": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 0.1,
                        "tooltip": "The amount of GPU memory to preserve.",
                    },
                ),
                "sampler": (["unipc_bh1", "unipc_bh2"], {"default": "unipc_bh1"}),
            },
            "optional": {
                "image_embeds": ("CLIP_VISION_OUTPUT",),
                "initial_samples": (
                    "LATENT",
                    {"tooltip": "init Latents to use for image2image variation"},
                ),
                "denoise_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "reference_latents": (
                    "REFERENCE_LATENT_LIST",
                    {"tooltip": "List of reference image latents for kisekaeichi mode (up to 8)"},
                ),
                "reference_image_embeds": (
                    "REFERENCE_EMBEDS_LIST",
                    {"tooltip": "List of reference image CLIP embeds for kisekaeichi mode (up to 8)"},
                ),
                "reference_masks": (
                    "REFERENCE_MASK_LIST",
                    {"tooltip": "List of reference masks for selective features (up to 8, optional)"},
                ),
                "target_index": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Target index for kisekaeichi mode (recommended: 1)",
                    },
                ),
                "control_index": (
                    "STRING",
                    {
                        "default": "0;10",
                        "tooltip": "Control indices separated by semicolon (preferred) or comma: 0;7;8;9;10 or 0,7,8,9,10 (default: 0;10)",
                    },
                ),
                "input_mask": (
                    "MASK",
                    {"tooltip": "Input mask for selective application"},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Single frame sampler with auto-enabled Kisekaeichi (style transfer) support for multiple references"

    def process(
        self,
        model,
        shift,
        positive,
        negative,
        latent_window_size,
        use_teacache,
        teacache_rel_l1_thresh,
        steps,
        cfg,
        guidance_scale,
        seed,
        sampler,
        gpu_memory_preservation,
        start_latent=None,
        image_embeds=None,
        initial_samples=None,
        denoise_strength=1.0,
        reference_latents=None,
        reference_image_embeds=None,
        reference_masks=None,
        target_index=1,
        control_index="0;10",
        input_mask=None,
    ):
        print("=== 1フレーム推論モード（musubi-tuner完全互換） ===")
        
        # Use control_index parameter directly (no more one_frame_inference)
        
        # Auto-enable kisekaeichi when reference inputs are provided
        use_kisekaeichi = reference_latents is not None and len(reference_latents) > 0
        if use_kisekaeichi:
            print("Kisekaeichi（着せ替え）モード有効 (自動検出)")
            print(f"target_index: {target_index}")
            print(f"control_index raw value: '{control_index}'")
            print(f"control_index type: {type(control_index)}")
            
            # Parse control indices from parameter (support both semicolon and comma for backward compatibility)
            if ';' in control_index:
                # New format (musubi-tuner compatible): semicolon-separated
                control_indices_list = [int(x.strip()) for x in control_index.split(';') if x.strip()]
            else:
                # Old format: comma-separated for backward compatibility
                control_indices_list = [int(x.strip()) for x in control_index.split(',') if x.strip()]
            
            if len(control_indices_list) == 0:
                control_indices_list = [0, 10]  # default
            print(f"parsed control_index: {control_indices_list}")

        transformer = model["transformer"]
        base_dtype = model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        # Latent処理
        if start_latent is not None:
            start_latent = start_latent["samples"] * vae_scaling_factor
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
        # Process reference latents if provided
        processed_reference_latents = []
        if use_kisekaeichi and reference_latents is not None:
            for i, ref_latent in enumerate(reference_latents):
                processed_latent = ref_latent["samples"] * vae_scaling_factor  
                processed_reference_latents.append(processed_latent)
                print(f"参照画像latent {i+1}: {processed_latent.shape}")

        print("start_latent", start_latent.shape)
        B, C, T, H, W = start_latent.shape

        # 画像エンベッディング処理
        if image_embeds is not None:
            start_image_encoder_last_hidden_state = image_embeds[
                "last_hidden_state"
            ].to(device, base_dtype)
        else:
            start_image_encoder_last_hidden_state = None

        # Process reference image embeds if provided
        processed_reference_embeds = []
        if use_kisekaeichi and reference_image_embeds is not None:
            for i, ref_embeds in enumerate(reference_image_embeds):
                embed_state = ref_embeds["last_hidden_state"].to(device, base_dtype)
                processed_reference_embeds.append(embed_state)
                print(f"参照画像 {i+1} のCLIP embeddingを設定しました")
        
        # Validate that latents and embeds have same count
        if use_kisekaeichi:
            if len(processed_reference_latents) != len(processed_reference_embeds):
                raise ValueError(f"Reference latents count ({len(processed_reference_latents)}) must match embeds count ({len(processed_reference_embeds)})")
            if len(processed_reference_latents) == 0:
                raise ValueError("No reference latents provided for kisekaeichi mode")

        # テキストエンベッディング処理
        llama_vec = positive[0][0].to(device, base_dtype)
        clip_l_pooler = positive[0][1]["pooled_output"].to(device, base_dtype)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(device, base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(device, base_dtype)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # シード設定
        rnd = torch.Generator("cpu").manual_seed(seed)

        # === 一つ目のコードに完全準拠した設定 ===

        # 1フレームモード固定設定
        sample_num_frames = 1
        total_latent_sections = 1
        latent_padding = 0
        latent_padding_size = latent_padding * latent_window_size  # 0

        # 一つ目のコードと同じインデックス構造
        indices = torch.arange(
            0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
        ).unsqueeze(0)
        split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]

        # latent_padding_sizeが0の場合の分割（一つ目のコードと完全同一）
        if latent_padding_size == 0:
            clean_latent_indices_pre = indices[:, 0:1]
            latent_indices = indices[:, 1 : 1 + latent_window_size]
            clean_latent_indices_post = indices[
                :, 1 + latent_window_size : 2 + latent_window_size
            ]
            clean_latent_2x_indices = indices[
                :, 2 + latent_window_size : 4 + latent_window_size
            ]
            clean_latent_4x_indices = indices[
                :, 4 + latent_window_size : 20 + latent_window_size
            ]
            blank_indices = torch.empty((1, 0), dtype=torch.long)
        else:
            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split(split_sizes, dim=1)

        # 一つ目のコードの重要な処理：1フレームモード時のone_frame_inference処理
        if sample_num_frames == 1:
            if use_kisekaeichi and len(processed_reference_latents) > 0:
                print("=== Kisekaeichi モード設定（完全版） ===")

                # 一つ目のコードのone_frame_inference処理を完全再現
                one_frame_inference = set()
                one_frame_inference.add(f"target_index={target_index}")
                # control_indexは後でcontrol_index_listから設定されるため、ここでは追加しない

                # musubi-tuner仕様：latent_indicesを初期化
                latent_indices = torch.zeros((1, 1), dtype=torch.int64)  # 1x1 latent index for target image
                latent_indices[:, 0] = latent_window_size  # musubi-tunerのデフォルト値
                
                # Apply target_index parameter (musubi-tuner仕様)
                latent_indices[:, 0] = target_index
                print(f"target_index設定 (musubi-tuner仕様): {target_index}")
                
                # この部分は削除（後でまとめて処理するため）

                # control_latentsのダミーを作成（一つ目のコードと同じ構造）
                control_latents = torch.zeros(
                    size=(1, 16, 1 + 2 + 16, H, W), dtype=torch.float32, device="cpu"
                )

                # musubi-tuner仕様：control_latentsを直接結合（マスク適用は後で実行）
                control_latents = []
                
                # 入力画像をcontrol_latentsに追加
                clean_latents_pre = start_latent.to(torch.float32).cpu()
                if len(clean_latents_pre.shape) < 5:
                    clean_latents_pre = clean_latents_pre.unsqueeze(2)
                control_latents.append(clean_latents_pre)
                
                # 参照画像をcontrol_latentsに追加
                for i, ref_latent in enumerate(processed_reference_latents):
                    clean_latent_post = ref_latent[:, :, 0:1, :, :].to(torch.float32).cpu()
                    control_latents.append(clean_latent_post)
                
                # musubi-tuner仕様：ゼロlatentをclean_latents_postとして追加
                control_latents.append(torch.zeros((1, 16, 1, H, W), dtype=torch.float32))
                print(f"Add zero latents as clean latents post for one frame inference.")
                
                # musubi-tuner仕様：control_latentsを直接結合
                clean_latents = torch.cat(control_latents, dim=2)  # (1, 16, num_control_images, H//8, W//8)

                # musubi-tuner仕様：clean_latent_indicesを設定
                clean_latent_indices = torch.zeros((1, len(control_latents)), dtype=torch.int64)
                if "no_post" not in []:  # one_frame_inferenceは空なので常にtrue
                    clean_latent_indices[:, -1] = 1 + latent_window_size  # default index for clean latents post
                
                # musubi-tuner仕様：control_indices_listをclean_latent_indicesに動的適用
                if control_indices_list:
                    print(f"control_indices動的適用開始: {control_indices_list}")
                    i = 0
                    while i < len(control_indices_list) and i < clean_latent_indices.shape[1]:
                        control_index = int(control_indices_list[i])
                        clean_latent_indices[:, i] = control_index
                        print(f"clean_latent_indices[:, {i}] = {control_index}")
                        i += 1
                    print(f"最終的なclean_latent_indices: {clean_latent_indices}")
                else:
                    print("control_indices_listが空のため、デフォルト設定を使用")
                
                # musubi-tuner仕様：マスク適用（clean_latents生成後）
                for i in range(len(control_latents)):
                    mask_image = None
                    
                    # 入力画像のマスク適用
                    if i == 0 and input_mask is not None:
                        print("入力画像マスクを適用中（musubi-tuner仕様）...")
                        try:
                            height_latent, width_latent = clean_latents.shape[-2:]
                            if isinstance(input_mask, torch.Tensor):
                                input_mask_tensor = input_mask
                            else:
                                input_mask_tensor = torch.from_numpy(input_mask)
                            
                            input_mask_resized = (
                                common_upscale(
                                    input_mask_tensor.unsqueeze(0).unsqueeze(0),
                                    width_latent,
                                    height_latent,
                                    "bilinear",
                                    "center",
                                )
                                .squeeze(0)
                                .squeeze(0)
                            )
                            mask_image = input_mask_resized.to(clean_latents.device)[None, None, :, :]
                            print("入力画像マスクを適用しました")
                        except Exception as e:
                            print(f"入力マスク適用エラー: {e}")
                    
                    # 参照画像のマスク適用
                    elif i > 0 and i < len(control_latents) - 1:  # -1はゼロlatentなので除外
                        ref_index = i - 1  # 最初が入力画像なので-1
                        if reference_masks is not None and ref_index < len(reference_masks):
                            print(f"参照画像 {ref_index+1} マスクを適用中（musubi-tuner仕様）...")
                            try:
                                height_latent, width_latent = clean_latents.shape[-2:]
                                reference_mask = reference_masks[ref_index]
                                
                                if isinstance(reference_mask, torch.Tensor):
                                    reference_mask_tensor = reference_mask
                                else:
                                    reference_mask_tensor = torch.from_numpy(reference_mask)
                                
                                reference_mask_resized = (
                                    common_upscale(
                                        reference_mask_tensor.unsqueeze(0).unsqueeze(0),
                                        width_latent,
                                        height_latent,
                                        "bilinear",
                                        "center",
                                    )
                                    .squeeze(0)
                                    .squeeze(0)
                                )
                                mask_image = reference_mask_resized.to(clean_latents.device)[None, None, :, :]
                                print(f"参照画像 {ref_index+1} マスクを適用しました")
                            except Exception as e:
                                print(f"参照マスク {ref_index+1} 適用エラー: {e}")
                    
                    # マスクを適用
                    if mask_image is not None:
                        clean_latents[:, :, i : i + 1, :, :] = clean_latents[:, :, i : i + 1, :, :] * mask_image

                # Always disable 2x/4x processing for kisekaeichi mode
                clean_latents_2x_param = None
                clean_latent_2x_indices = None
                clean_latents_2x = None
                
                clean_latents_4x_param = None
                clean_latent_4x_indices = None
                clean_latents_4x = None

                print("Kisekaeichi: 2x/4xインデックスを無効化しました")

                # 画像エンベッディングの処理（musubi-tuner仕様：セクション毎の個別処理）
                # musubi-tunerではセクション毎に個別のimage_encoder_last_hidden_stateを使用
                # 複数リファレンス時は最初の参照画像のエンベッディングを使用（musubi-tuner互換）
                if len(processed_reference_embeds) > 0:
                    # musubi-tuner互換：最初の参照画像エンベッディングを使用
                    image_encoder_last_hidden_state = processed_reference_embeds[0]
                    print(f"musubi-tuner互換：最初の参照画像エンベッディングを使用 (参照画像数: {len(processed_reference_embeds)})")
                else:
                    image_encoder_last_hidden_state = start_image_encoder_last_hidden_state

                print(f"Kisekaeichi設定完了（musubi-tuner完全互換）:")
                print(f"  - clean_latents.shape: {clean_latents.shape} (入力+参照{len(processed_reference_latents)}個+ゼロlatent)")
                print(f"  - latent_indices: {latent_indices} (初期値: {latent_window_size} -> target: {target_index})")
                print(f"  - clean_latent_indices: {clean_latent_indices} (control_indices適用済み)")
                print(f"  - sample_num_frames: {sample_num_frames}")
                print(f"  - control_indices適用: {control_indices_list}")
                print(f"  - control_latents数: {len(control_latents)}")
                print(f"  - マスク適用: clean_latents生成後")
                print(f"  - 2x/4x無効化: True")

            else:
                # 通常モード（参照画像なし）
                # control_indicesは通常モードでは使用しない
                all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
                latent_indices = all_indices[:, -1:]

                clean_latents_pre = start_latent.to(torch.float32).cpu()
                if len(clean_latents_pre.shape) < 5:
                    clean_latents_pre = clean_latents_pre.unsqueeze(2)

                clean_latents_post = torch.zeros_like(clean_latents_pre)
                clean_latents = torch.cat(
                    [clean_latents_pre, clean_latents_post], dim=2
                )
                clean_latent_indices = torch.cat(
                    [clean_latent_indices_pre, clean_latent_indices_post], dim=1
                )

                # 通常モードでのインデックス調整
                clean_latent_indices = torch.tensor(
                    [[0]],
                    dtype=clean_latent_indices.dtype,
                    device=clean_latent_indices.device,
                )
                clean_latents = clean_latents[:, :, :1, :, :]

                clean_latents_2x_param = None
                clean_latents_4x_param = None
                clean_latent_2x_indices = None
                clean_latent_4x_indices = None

                # 2x, 4x latentsの設定も無効化
                clean_latents_2x = None
                clean_latents_4x = None

                print("Kisekaeichi: 2x/4xインデックスを無効化しました")

                image_encoder_last_hidden_state = start_image_encoder_last_hidden_state

                print("通常モード設定:")
                print(f"  - clean_latents.shape: {clean_latents.shape}")
                print(f"  - latent_indices: {latent_indices}")
                print(f"  - clean_latent_indices: {clean_latent_indices}")

        # 初期サンプルの処理
        input_init_latents = None
        if initial_samples is not None:
            input_init_latents = initial_samples[:, :, 0:1, :, :].to(device)
            print("初期サンプルを設定しました")

        # ComfyUI用のセットアップ
        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        patcher = comfy.model_patcher.ModelPatcher(
            comfy_model, device, torch.device("cpu")
        )
        from latent_preview import prepare_callback

        callback = prepare_callback(patcher, steps)

        # モデルをGPUに移動
        move_model_to_device_with_memory_preservation(
            transformer,
            target_device=device,
            preserved_memory_gb=gpu_memory_preservation,
        )

        # TeaCacheの設定
        if use_teacache:
            transformer.initialize_teacache(
                enable_teacache=True,
                num_steps=steps,
                rel_l1_thresh=teacache_rel_l1_thresh,
            )
        else:
            transformer.initialize_teacache(enable_teacache=False)

        print("=== サンプリング開始 ===")
        print(f"sample_num_frames: {sample_num_frames}")
        print(f"clean_latents使用フレーム数: {clean_latents.shape[2]}")
        print(f"clean_latent_2x_indices: {clean_latent_2x_indices}")
        print(f"clean_latent_4x_indices: {clean_latent_4x_indices}")

        with torch.autocast(
            device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True
        ):
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler=sampler,
                initial_latent=input_init_latents,
                strength=denoise_strength,
                width=W * 8,
                height=H * 8,
                frames=sample_num_frames,  # 1フレーム固定
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

        # クリーンアップ
        transformer.to(offload_device)
        mm.soft_empty_cache()

        # 出力処理
        if use_kisekaeichi:
            mode_info = f"Kisekaeichi（musubi-tuner完全互換、参照{len(processed_reference_latents)}個）"
        else:
            mode_info = "通常"
        print(
            f"=== 1フレーム生成完了 ({mode_info}モード): {generated_latents.shape} ==="
        )

        return ({"samples": generated_latents / vae_scaling_factor},)


class ReferenceLatentList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "reference_1": ("LATENT", {"tooltip": "Reference latent 1"}),
                "reference_2": ("LATENT", {"tooltip": "Reference latent 2"}),
                "reference_3": ("LATENT", {"tooltip": "Reference latent 3"}),
                "reference_4": ("LATENT", {"tooltip": "Reference latent 4"}),
                "reference_5": ("LATENT", {"tooltip": "Reference latent 5"}),
                "reference_6": ("LATENT", {"tooltip": "Reference latent 6"}),
                "reference_7": ("LATENT", {"tooltip": "Reference latent 7"}),
                "reference_8": ("LATENT", {"tooltip": "Reference latent 8"}),
            },
        }

    RETURN_TYPES = ("REFERENCE_LATENT_LIST",)
    RETURN_NAMES = ("reference_latents",)
    FUNCTION = "create_list"
    CATEGORY = "FramePackWrapper/References"
    DESCRIPTION = "Combine up to 8 reference latents into a list for multi-reference Kisekaeichi mode"

    def create_list(self, **kwargs):
        reference_list = []
        for i in range(1, 9):
            ref_key = f"reference_{i}"
            if ref_key in kwargs and kwargs[ref_key] is not None:
                reference_list.append(kwargs[ref_key])
        
        if len(reference_list) == 0:
            return (None,)
        
        print(f"Created reference latent list with {len(reference_list)} items")
        return (reference_list,)


class ReferenceEmbedsList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "embeds_1": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 1"}),
                "embeds_2": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 2"}),
                "embeds_3": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 3"}),
                "embeds_4": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 4"}),
                "embeds_5": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 5"}),
                "embeds_6": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 6"}),
                "embeds_7": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 7"}),
                "embeds_8": ("CLIP_VISION_OUTPUT", {"tooltip": "Reference embeds 8"}),
            },
        }

    RETURN_TYPES = ("REFERENCE_EMBEDS_LIST",)
    RETURN_NAMES = ("reference_embeds",)
    FUNCTION = "create_list"
    CATEGORY = "FramePackWrapper/References"
    DESCRIPTION = "Combine up to 8 reference CLIP vision embeddings into a list for multi-reference Kisekaeichi mode"

    def create_list(self, **kwargs):
        embeds_list = []
        for i in range(1, 9):
            embeds_key = f"embeds_{i}"
            if embeds_key in kwargs and kwargs[embeds_key] is not None:
                embeds_list.append(kwargs[embeds_key])
        
        if len(embeds_list) == 0:
            return (None,)
        
        print(f"Created reference embeds list with {len(embeds_list)} items")
        return (embeds_list,)


class ReferenceMaskList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "mask_1": ("MASK", {"tooltip": "Reference mask 1"}),
                "mask_2": ("MASK", {"tooltip": "Reference mask 2"}),
                "mask_3": ("MASK", {"tooltip": "Reference mask 3"}),
                "mask_4": ("MASK", {"tooltip": "Reference mask 4"}),
                "mask_5": ("MASK", {"tooltip": "Reference mask 5"}),
                "mask_6": ("MASK", {"tooltip": "Reference mask 6"}),
                "mask_7": ("MASK", {"tooltip": "Reference mask 7"}),
                "mask_8": ("MASK", {"tooltip": "Reference mask 8"}),
            },
        }

    RETURN_TYPES = ("REFERENCE_MASK_LIST",)
    RETURN_NAMES = ("reference_masks",)
    FUNCTION = "create_list"
    CATEGORY = "FramePackWrapper/References"
    DESCRIPTION = "Combine up to 8 reference masks into a list for multi-reference Kisekaeichi mode (optional)"

    def create_list(self, **kwargs):
        mask_list = []
        for i in range(1, 9):
            mask_key = f"mask_{i}"
            if mask_key in kwargs and kwargs[mask_key] is not None:
                mask_list.append(kwargs[mask_key])
        
        if len(mask_list) == 0:
            return (None,)
        
        print(f"Created reference mask list with {len(mask_list)} items")
        return (mask_list,)


NODE_CLASS_MAPPINGS = {
    "FramePackSingleFrameSampler": FramePackSingleFrameSampler,
    "ReferenceLatentList": ReferenceLatentList,
    "ReferenceEmbedsList": ReferenceEmbedsList,
    "ReferenceMaskList": ReferenceMaskList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePackSingleFrameSampler": "FramePack Single Frame Sampler",
    "ReferenceLatentList": "Reference Latent List",
    "ReferenceEmbedsList": "Reference Embeds List", 
    "ReferenceMaskList": "Reference Mask List",
}
