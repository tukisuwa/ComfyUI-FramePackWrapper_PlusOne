# 動作確認のための Makefile
# 1. .env を作成
# 2. extra_model_paths.yaml を作成
# 3. make init && make run

include .env	# MODEL_PATH, TMP

MODEL_PATH ?= ComfyUI/models
TMP ?= /tmp

.PHONY: run

run:
	ComfyUI/.venv/bin/python ComfyUI/main.py --listen --port 58188 --fast --extra-model-paths-config extra_model_paths.yaml

ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI.git

ComfyUI/.venv: ComfyUI
	test -d $@ || python3 -m venv $@
	ComfyUI/.venv/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
	ComfyUI/.venv/bin/pip install -r ComfyUI/requirements.txt

custom_nodes = \
	ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_PlusOne \
	ComfyUI/custom_nodes/comfyui-get-meta \
	ComfyUI/custom_nodes/ComfyUI-KJNodes \
	ComfyUI/custom_nodes/ComfyUI-LogicUtils \
	ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite \
	ComfyUI/custom_nodes/ComfyUI_essentials

ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_PlusOne: REPO = .
ComfyUI/custom_nodes/ComfyUI-FramePackWrapper_PlusOne: COMMIT = $(shell git rev-parse HEAD)
ComfyUI/custom_nodes/comfyui-get-meta: REPO = https://github.com/shinich39/comfyui-get-meta
ComfyUI/custom_nodes/comfyui-get-meta: COMMIT = b1af205fed09a3a4e2257f208fee2c53ad27a96e
ComfyUI/custom_nodes/ComfyUI-KJNodes: REPO = https://github.com/kijai/ComfyUI-KJNodes
ComfyUI/custom_nodes/ComfyUI-KJNodes: COMMIT = 5dcda71011870278c35d92ff77a677ed2e538f2d
ComfyUI/custom_nodes/ComfyUI-LogicUtils: REPO = https://github.com/aria1th/ComfyUI-LogicUtils
ComfyUI/custom_nodes/ComfyUI-LogicUtils: COMMIT = 60f8f1187c66ee544e09a85303e4140cf0bd0ff2
ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite: REPO = https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite: COMMIT = a7ce59e381934733bfae03b1be029756d6ce936d
ComfyUI/custom_nodes/ComfyUI_essentials: REPO = https://github.com/cubiq/ComfyUI_essentials
ComfyUI/custom_nodes/ComfyUI_essentials: COMMIT = 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9

$(custom_nodes): ComfyUI
	git -C $@ rev-parse HEAD || git clone $(REPO) $@
	git -C $@ checkout $(COMMIT)
	test -f $@/requirements.txt && ComfyUI/.venv/bin/pip install -r $@/requirements.txt || true

models = \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_fp8_e4m3fn.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_bf16.safetensors \
	$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors \
	$(MODEL_PATH)/text_encoders/clip_l.safetensors \
	$(MODEL_PATH)/text_encoders/llava_llama3_fp16.safetensors \
	$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors \
	$(MODEL_PATH)/vae/hunyuan_video_vae_bf16.safetensors \
	$(MODEL_PATH)/loras/framepack/body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors \
	$(MODEL_PATH)/loras/framepack/Apose_V7_dim4.safetensors \
	$(MODEL_PATH)/loras/framepack/fp-1f-chibi-1024.safetensors

## diffusion_models
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_fp8_e4m3fn.safetensors: REPO = Kijai/HunyuanVideo_comfy
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_fp8_e4m3fn.safetensors: FILE = FramePackI2V_HY_fp8_e4m3fn.safetensors
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_bf16.safetensors: REPO = Kijai/HunyuanVideo_comfy
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY_bf16.safetensors: FILE = FramePackI2V_HY_bf16.safetensors
## clip_vision
$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors: REPO = Comfy-Org/sigclip_vision_384
$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors: FILE = sigclip_vision_patch14_384.safetensors

## text_encoders
$(MODEL_PATH)/text_encoders/clip_l.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/clip_l.safetensors: FILE = split_files/text_encoders/clip_l.safetensors
$(MODEL_PATH)/text_encoders/llava_llama3_fp16.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/llava_llama3_fp16.safetensors: FILE = split_files/text_encoders/llava_llama3_fp16.safetensors
$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors: FILE = split_files/text_encoders/llava_llama3_fp8_scaled.safetensors

## vae
$(MODEL_PATH)/vae/hunyuan_video_vae_bf16.safetensors: REPO = Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/vae/hunyuan_video_vae_bf16.safetensors: FILE = split_files/vae/hunyuan_video_vae_bf16.safetensors

## loras
$(MODEL_PATH)/loras/framepack/body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors: REPO = tori29umai/FramePack_LoRA
$(MODEL_PATH)/loras/framepack/body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors: FILE = body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors
$(MODEL_PATH)/loras/framepack/Apose_V7_dim4.safetensors: REPO = tori29umai/FramePack_LoRA
$(MODEL_PATH)/loras/framepack/Apose_V7_dim4.safetensors: FILE = Apose_V7_dim4.safetensors
$(MODEL_PATH)/loras/framepack/fp-1f-chibi-1024.safetensors: REPO = kohya-ss/misc-models
$(MODEL_PATH)/loras/framepack/fp-1f-chibi-1024.safetensors: FILE = fp-1f-chibi-1024.safetensors

$(models):
	uvx --from "huggingface_hub[cli]" huggingface-cli download $(REPO) $(FILE) --local-dir $(TMP)/$(REPO)
	mkdir -p $(dir $@)
	mv $(TMP)/$(REPO)/$(FILE) $@

init: ComfyUI/.venv $(custom_nodes) $(models)

patch-tori29umai0123:
	git remote | grep upstream || git remote add upstream https://github.com/tori29umai0123/ComfyUI-FramePackWrapper_PlusOne.git
	git fetch upstream
	git checkout upstream/main -- .
	@echo "tori29umai0123/ComfyUI-FramePackWrapper_PlusOne をパッチしました。Staged Changesを確認してコミットしてください。ファイルに差異がある場合、削除を含めて全て先方が優先されます。"
