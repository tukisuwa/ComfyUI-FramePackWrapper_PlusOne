import numpy as np
from PIL import Image
import logging
import math
from tqdm import tqdm
from latent_preview import prepare_callback

import torch

logger = logging.getLogger(__name__)

import comfy.model_base
import comfy.model_management as mm

from .diffusers_helper.memory import move_model_to_device_with_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .nodes import HyVideoModel, HyVideoModelConfig

# Constants
VAE_SCALING_FACTOR = 0.476986
DEFAULT_BLEND_ALPHA = 0.5
OVERLAP_RATIO = 0.5  # 50% オーバーラップ


class TileProcessor:
    """50%オーバーラップを持つ2次元画像タイル処理を担当するクラス"""
    
    def __init__(self, tile_size: int, tile_resized: int):
        self.tile_size = tile_size
        self.tile_resized = tile_resized
        self.upscale_factor = float(tile_resized) / float(tile_size)
        self.step_size = int(tile_size * (1 - OVERLAP_RATIO))  # 50%オーバーラップのステップサイズ
    
    def split_image_to_tiles(self, image):
        """画像を50%オーバーラップでタイルに分割し、白パディングを追加して指定サイズにリサイズ"""
        pil_image = self._convert_to_pil(image)
        original_width, original_height = pil_image.size
        
        # オーバーラップを考慮したタイル数を計算
        tiles_x = self._calculate_tile_count(original_width)
        tiles_y = self._calculate_tile_count(original_height)
        
        self._log_split_info(original_width, original_height, tiles_x, tiles_y)
        
        metadata = self._create_metadata(original_width, original_height, tiles_x, tiles_y)
        tiles = self._extract_overlapping_tiles(pil_image, tiles_x, tiles_y, metadata)
        
        return tiles, metadata
    
    def _calculate_tile_count(self, dimension: int) -> int:
        """オーバーラップを考慮したタイル数を計算"""
        if dimension <= self.tile_size:
            return 1
        # 最後のタイルが完全に収まるように計算
        return math.ceil((dimension - self.tile_size) / self.step_size) + 1
    
    def _convert_to_pil(self, image):
        """ComfyUIのIMAGE形式からPIL形式に変換"""
        if isinstance(image, torch.Tensor):
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(image_np)
        return image
    
    def _log_split_info(self, width: int, height: int, tiles_x: int, tiles_y: int):
        """分割情報をログ出力"""
        print(f"=== 50%オーバーラップ画像分割情報 ===")
        print(f"元画像サイズ: {width} × {height}")
        print(f"分割タイルサイズ: {self.tile_size} × {self.tile_size}")
        print(f"リサイズ後タイルサイズ: {self.tile_resized} × {self.tile_resized}")
        print(f"アップスケール倍率: {self.upscale_factor:.2f}x")
        print(f"ステップサイズ: {self.step_size} (50%オーバーラップ)")
        print(f"タイル数: {tiles_x} × {tiles_y} = {tiles_x * tiles_y}個")
    
    def _create_metadata(self, width: int, height: int, tiles_x: int, tiles_y: int) -> dict:
        """メタデータを作成"""
        return {
            "original_size": [width, height],
            "tile_size": self.tile_size,
            "tile_resized": self.tile_resized,
            "upscale_factor": self.upscale_factor,
            "step_size": self.step_size,
            "overlap_ratio": OVERLAP_RATIO,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "tiles": []
        }
    
    def _extract_overlapping_tiles(self, pil_image: Image.Image, tiles_x: int, tiles_y: int, metadata: dict) -> list:
        """50%オーバーラップでタイルを切り出す"""
        tiles = []
        tile_index = 0
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                tile_tensor, tile_info = self._process_overlapping_tile(
                    pil_image, x, y, tile_index
                )
                tiles.append(tile_tensor)
                metadata["tiles"].append(tile_info)
                tile_index += 1
        
        return tiles
    
    def _process_overlapping_tile(self, pil_image: Image.Image, x: int, y: int, tile_index: int):
        """50%オーバーラップを考慮した単一タイルを処理"""
        # オーバーラップを考慮した切り出し位置を計算
        left = x * self.step_size
        top = y * self.step_size
        right = min(left + self.tile_size, pil_image.width)
        bottom = min(top + self.tile_size, pil_image.height)
        
        # 画像端で調整（右端・下端のタイルが小さくなる場合）
        if right - left < self.tile_size:
            left = max(0, right - self.tile_size)
        if bottom - top < self.tile_size:
            top = max(0, bottom - self.tile_size)
        
        # タイルを切り出し
        tile = pil_image.crop((left, top, right, bottom))
        tile_width, tile_height = tile.size
        
        # パディング処理
        padding_needed = (tile_width < self.tile_size or tile_height < self.tile_size)
        if padding_needed:
            tile = self._add_white_padding(tile)
        
        # アップスケール
        upscaled_tile = tile.resize((self.tile_resized, self.tile_resized), Image.LANCZOS)
        
        # テンソル変換
        tile_tensor = torch.from_numpy(np.array(upscaled_tile)).float() / 255.0
        tile_tensor = tile_tensor.unsqueeze(0)  # バッチ次元を追加
        
        # オーバーラップ領域の情報を計算
        overlap_info = self._calculate_overlap_info(x, y, left, top, right, bottom, pil_image.size)
        
        # メタデータ作成
        tile_info = {
            "index": tile_index,
            "grid_position": [x, y],
            "crop_box": [left, top, right, bottom],
            "original_tile_size": [tile_width, tile_height],
            "padding_needed": padding_needed,
            "padding_right": self.tile_size - tile_width if padding_needed else 0,
            "padding_bottom": self.tile_size - tile_height if padding_needed else 0,
            "overlap_info": overlap_info
        }
        
        print(f"タイル {tile_index}: グリッド({x},{y}) 切り出し({left},{top},{right},{bottom}) "
              f"オーバーラップ:{overlap_info}")
        
        return tile_tensor, tile_info
    
    def _calculate_overlap_info(self, grid_x: int, grid_y: int, left: int, top: int, 
                              right: int, bottom: int, image_size: tuple) -> dict:
        """オーバーラップ領域の情報を計算"""
        image_width, image_height = image_size
        overlap_info = {
            "has_left_overlap": grid_x > 0,
            "has_right_overlap": right < image_width and grid_x < self._calculate_tile_count(image_width) - 1,
            "has_top_overlap": grid_y > 0,
            "has_bottom_overlap": bottom < image_height and grid_y < self._calculate_tile_count(image_height) - 1,
        }
        
        # オーバーラップピクセル数を計算
        overlap_pixels = int(self.tile_size * OVERLAP_RATIO)
        overlap_info.update({
            "left_overlap_pixels": overlap_pixels if overlap_info["has_left_overlap"] else 0,
            "right_overlap_pixels": overlap_pixels if overlap_info["has_right_overlap"] else 0,
            "top_overlap_pixels": overlap_pixels if overlap_info["has_top_overlap"] else 0,
            "bottom_overlap_pixels": overlap_pixels if overlap_info["has_bottom_overlap"] else 0,
        })
        
        return overlap_info
    
    def _add_white_padding(self, tile: Image.Image) -> Image.Image:
        """白パディングを追加"""
        padded_tile = Image.new('RGB', (self.tile_size, self.tile_size), color='white')
        padded_tile.paste(tile, (0, 0))
        return padded_tile


class ImageReconstructor:
    """50%オーバーラップしたタイルから画像を再構成するクラス"""
    
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.upscale_factor = metadata["upscale_factor"]
        self.tile_size = metadata["tile_size"]
        self.step_size = metadata["step_size"]
        self.overlap_ratio = metadata["overlap_ratio"]
    
    def reconstruct_image(self, processed_tiles: list):
        """50%オーバーラップしたタイルから画像を再構成"""
        final_width, final_height = self._calculate_final_size()
        self._log_reconstruction_info(final_width, final_height, len(processed_tiles))
        
        # 最終画像とブレンドウェイトマップを初期化
        final_image = np.zeros((final_height, final_width, 3), dtype=np.float32)
        weight_map = np.zeros((final_height, final_width), dtype=np.float32)
        
        # 処理されたタイル画像を準備
        processed_tile_images = self._preprocess_tiles(processed_tiles)
        
        # オーバーラップブレンドでタイルを配置
        self._place_tiles_with_overlap_blend(final_image, weight_map, processed_tile_images)
        
        # 正規化して最終画像を作成
        final_image = self._normalize_blended_image(final_image, weight_map)
        
        return self._convert_to_tensor(Image.fromarray(final_image.astype(np.uint8)))
    
    def _calculate_final_size(self) -> tuple:
        """最終画像サイズを計算"""
        original_width, original_height = self.metadata["original_size"]
        final_width = int(original_width * self.upscale_factor)
        final_height = int(original_height * self.upscale_factor)
        return final_width, final_height
    
    def _log_reconstruction_info(self, width: int, height: int, tile_count: int):
        """再構成情報をログ出力"""
        print(f"=== 50%オーバーラップ画像再構成情報 ===")
        print(f"最終画像サイズ: {width} × {height} ({self.upscale_factor:.2f}x)")
        print(f"処理タイル数: {tile_count}")
        print(f"オーバーラップ比率: {self.overlap_ratio*100:.0f}%")
    
    def _preprocess_tiles(self, processed_tiles: list) -> list:
        """タイル画像を前処理"""
        processed_tile_images = []
        for i, (tile_tensor, tile_info) in enumerate(zip(processed_tiles, self.metadata["tiles"])):
            tile_image = self._tensor_to_pil(tile_tensor)
            
            # パディング部分を除去
            if tile_info["padding_needed"]:
                tile_image = self._remove_padding(tile_image, tile_info)
            
            processed_tile_images.append((tile_image, tile_info))
        
        return processed_tile_images
    
    def _tensor_to_pil(self, tile_tensor) -> Image.Image:
        """テンソルをPIL画像に変換"""
        if isinstance(tile_tensor, torch.Tensor):
            tile_np = (tile_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(tile_np)
        return tile_tensor
    
    def _remove_padding(self, tile_image: Image.Image, tile_info: dict) -> Image.Image:
        """パディング部分を除去"""
        original_tile_width, original_tile_height = tile_info["original_tile_size"]
        content_width = int(original_tile_width * self.upscale_factor)
        content_height = int(original_tile_height * self.upscale_factor)
        return tile_image.crop((0, 0, content_width, content_height))
    
    def _place_tiles_with_overlap_blend(self, final_image: np.ndarray, weight_map: np.ndarray, 
                                       processed_tile_images: list):
        """オーバーラップ領域でブレンドしながらタイルを配置"""
        for i, (tile_image, tile_info) in enumerate(processed_tile_images):
            grid_x, grid_y = tile_info["grid_position"]
            crop_box = tile_info["crop_box"]
            overlap_info = tile_info["overlap_info"]
            
            # 元の切り出し位置をアップスケール倍率で拡大
            left, top, right, bottom = crop_box
            paste_left = int(left * self.upscale_factor)
            paste_top = int(top * self.upscale_factor)
            paste_right = int(right * self.upscale_factor)
            paste_bottom = int(bottom * self.upscale_factor)
            
            # タイル画像を numpy 配列に変換
            tile_array = np.array(tile_image, dtype=np.float32)
            tile_height, tile_width = tile_array.shape[:2]
            
            # 最終画像の境界内に収まるように調整
            paste_right = min(paste_right, final_image.shape[1])
            paste_bottom = min(paste_bottom, final_image.shape[0])
            actual_width = paste_right - paste_left
            actual_height = paste_bottom - paste_top
            
            # タイルサイズが配置領域と異なる場合は調整
            if actual_width != tile_width or actual_height != tile_height:
                if actual_width > 0 and actual_height > 0:
                    tile_array = tile_array[:actual_height, :actual_width]
                else:
                    print(f"警告: タイル {i} の配置領域が無効です")
                    continue
            
            # ブレンド用のウェイトマスクを作成
            tile_weight = self._create_blend_weight_mask(
                tile_array.shape[:2], overlap_info, crop_box, self.upscale_factor
            )
            
            # 加算ブレンド方式で配置
            final_image[paste_top:paste_bottom, paste_left:paste_right] += \
                tile_array * tile_weight[..., np.newaxis]
            weight_map[paste_top:paste_bottom, paste_left:paste_right] += tile_weight
            
            print(f"タイル {i} 配置: 元座標({left},{top},{right},{bottom}) -> "
                  f"配置座標({paste_left},{paste_top},{paste_right},{paste_bottom}) "
                  f"サイズ({actual_width}×{actual_height})")
    
    def _create_blend_weight_mask(self, tile_shape: tuple, overlap_info: dict, 
                                crop_box: list, upscale_factor: float) -> np.ndarray:
        """オーバーラップ領域用のウェイトマスクを作成"""
        height, width = tile_shape
        weight_mask = np.ones((height, width), dtype=np.float32)
        
        # オーバーラップピクセル数を計算（アップスケール後）
        overlap_pixels = int(self.tile_size * self.overlap_ratio * upscale_factor)
        
        # 実際の切り出しボックスを使ってオーバーラップ領域を正確に計算
        left, top, right, bottom = crop_box
        
        # 左端のフェード（左にオーバーラップがある場合）
        if overlap_info["has_left_overlap"]:
            # 実際のオーバーラップ開始位置を計算
            overlap_start_x = left % self.step_size
            if overlap_start_x > 0:
                fade_width = min(int(overlap_start_x * upscale_factor), width)
                if fade_width > 0:
                    fade_weights = np.linspace(0, 1, fade_width)
                    weight_mask[:, :fade_width] *= fade_weights[np.newaxis, :]
        
        # 右端のフェード（右にオーバーラップがある場合）
        if overlap_info["has_right_overlap"]:
            # 右端のオーバーラップ幅を計算
            next_tile_start = (left // self.step_size + 1) * self.step_size
            if right > next_tile_start:
                overlap_width = right - next_tile_start
                fade_width = min(int(overlap_width * upscale_factor), width)
                if fade_width > 0:
                    fade_weights = np.linspace(1, 0, fade_width)
                    weight_mask[:, -fade_width:] *= fade_weights[np.newaxis, :]
        
        # 上端のフェード（上にオーバーラップがある場合）
        if overlap_info["has_top_overlap"]:
            # 実際のオーバーラップ開始位置を計算
            overlap_start_y = top % self.step_size
            if overlap_start_y > 0:
                fade_height = min(int(overlap_start_y * upscale_factor), height)
                if fade_height > 0:
                    fade_weights = np.linspace(0, 1, fade_height)
                    weight_mask[:fade_height, :] *= fade_weights[:, np.newaxis]
        
        # 下端のフェード（下にオーバーラップがある場合）
        if overlap_info["has_bottom_overlap"]:
            # 下端のオーバーラップ高さを計算
            next_tile_start = (top // self.step_size + 1) * self.step_size
            if bottom > next_tile_start:
                overlap_height = bottom - next_tile_start
                fade_height = min(int(overlap_height * upscale_factor), height)
                if fade_height > 0:
                    fade_weights = np.linspace(1, 0, fade_height)
                    weight_mask[-fade_height:, :] *= fade_weights[:, np.newaxis]
        
        return weight_mask
    
    def _normalize_blended_image(self, final_image: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
        """ブレンドされた画像を正規化"""
        # ゼロ除算を避けるため、重みが0の部分には小さな値を設定
        weight_map = np.maximum(weight_map, 1e-8)
        
        # 正規化
        final_image = final_image / weight_map[..., np.newaxis]
        
        # 値の範囲を0-255にクランプ
        final_image = np.clip(final_image, 0, 255)
        
        return final_image
    
    def _convert_to_tensor(self, final_image: Image.Image) -> torch.Tensor:
        """PIL画像をComfyUI形式のテンソルに変換"""
        final_tensor = torch.from_numpy(np.array(final_image)).float() / 255.0
        return final_tensor.unsqueeze(0)  # バッチ次元を追加


class VAEProcessor:
    """VAE処理を担当するクラス"""
    
    @staticmethod
    def tiled_decode(vae, latents: torch.Tensor, tile_size: int = 256, 
                    overlap: int = 64, temporal_size: int = 64, 
                    temporal_overlap: int = 8) -> torch.Tensor:
        """タイル化VAEデコード"""
        B, C, T, H, W = latents.shape
        device = latents.device
        dtype = latents.dtype
        
        # 出力テンソルを初期化 (VAEは8倍アップスケール)
        output_h, output_w = H * 8, W * 8
        output = torch.zeros(B, T, output_h, output_w, 3, device=device, dtype=dtype)
        
        # タイル分割数を計算
        h_tiles, w_tiles, t_tiles = VAEProcessor._calculate_tile_counts(
            H, W, T, tile_size, overlap, temporal_size, temporal_overlap
        )
        
        print(f"タイル化デコード: H={h_tiles}, W={w_tiles}, T={t_tiles}")
        
        # タイル処理実行
        VAEProcessor._process_tiles(
            vae, latents, output, h_tiles, w_tiles, t_tiles,
            tile_size, overlap, temporal_size, temporal_overlap, device, dtype
        )
        
        return output
    
    @staticmethod
    def _calculate_tile_counts(H: int, W: int, T: int, tile_size: int, 
                             overlap: int, temporal_size: int, temporal_overlap: int) -> tuple:
        """タイル分割数を計算"""
        h_tiles = max(1, (H - overlap) // (tile_size - overlap) + 
                     (1 if (H - overlap) % (tile_size - overlap) > 0 else 0))
        w_tiles = max(1, (W - overlap) // (tile_size - overlap) + 
                     (1 if (W - overlap) % (tile_size - overlap) > 0 else 0))
        t_tiles = max(1, (T - temporal_overlap) // (temporal_size - temporal_overlap) + 
                     (1 if (T - temporal_overlap) % (temporal_size - temporal_overlap) > 0 else 0))
        return h_tiles, w_tiles, t_tiles
    
    @staticmethod
    def _process_tiles(vae, latents: torch.Tensor, output: torch.Tensor,
                      h_tiles: int, w_tiles: int, t_tiles: int,
                      tile_size: int, overlap: int, temporal_size: int, 
                      temporal_overlap: int, device, dtype):
        """実際のタイル処理を実行"""
        B, C, T, H, W = latents.shape
        
        for t_idx in range(t_tiles):
            t_start, t_end = VAEProcessor._calculate_range(
                t_idx, temporal_size, temporal_overlap, T
            )
            
            for h_idx in range(h_tiles):
                h_start, h_end = VAEProcessor._calculate_range(
                    h_idx, tile_size, overlap, H
                )
                
                for w_idx in range(w_tiles):
                    w_start, w_end = VAEProcessor._calculate_range(
                        w_idx, tile_size, overlap, W
                    )
                    
                    # タイルを切り出してデコード
                    tile_latent = latents[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                    
                    try:
                        with torch.autocast(device_type=mm.get_autocast_device(device), 
                                          dtype=dtype, enabled=True):
                            decoded_tile = vae.decode(tile_latent)
                        
                        # 出力に配置（オーバーラップ処理含む）
                        VAEProcessor._place_decoded_tile(
                            output, decoded_tile, t_start, t_end, 
                            h_start, h_end, w_start, w_end, h_idx, w_idx
                        )
                        
                    except Exception as e:
                        print(f"タイル({t_idx},{h_idx},{w_idx})のデコードに失敗: {e}")
                        continue
    
    @staticmethod
    def _calculate_range(idx: int, size: int, overlap: int, max_size: int) -> tuple:
        """範囲を計算"""
        start = idx * (size - overlap)
        end = min(start + size, max_size)
        start = max(0, end - size)  # 境界調整
        return start, end
    
    @staticmethod
    def _place_decoded_tile(output: torch.Tensor, decoded_tile: torch.Tensor,
                           t_start: int, t_end: int, h_start: int, h_end: int,
                           w_start: int, w_end: int, h_idx: int, w_idx: int):
        """デコードされたタイルを出力に配置"""
        out_t_start, out_t_end = t_start, t_end
        out_h_start, out_h_end = h_start * 8, h_end * 8
        out_w_start, out_w_end = w_start * 8, w_end * 8
        
        # オーバーラップ処理
        if h_idx > 0 or w_idx > 0:
            existing = output[:, out_t_start:out_t_end, out_h_start:out_h_end, out_w_start:out_w_end]
            if existing.sum() > 0:  # 既存データがある場合のみブレンド
                output[:, out_t_start:out_t_end, out_h_start:out_h_end, out_w_start:out_w_end] = \
                    DEFAULT_BLEND_ALPHA * existing + (1 - DEFAULT_BLEND_ALPHA) * decoded_tile
            else:
                output[:, out_t_start:out_t_end, out_h_start:out_h_end, out_w_start:out_w_end] = decoded_tile
        else:
            # 最初のタイルはそのまま配置
            output[:, out_t_start:out_t_end, out_h_start:out_h_end, out_w_start:out_w_end] = decoded_tile


class FramePackSingleFrameResizeSampler:
    """メインのサンプラークラス"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "input_image": ("IMAGE", {"tooltip": "Input image to be upscaled"}),
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
                "tile_size": ("INT", {"default": 256, "min": 256, "max": 1024, "step": 64, "tooltip": "Size of each tile for initial split"}),
                "tile_resized": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "tooltip": "Size to resize each tile to before processing"}),
                "vae_tile_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 64, "tooltip": "Size of each VAE tile for decode"}),
                "vae_overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8, "tooltip": "Overlap between VAE tiles"}),
                "vae_temporal_size": ("INT", {"default": 64, "min": 1, "max": 128, "step": 1, "tooltip": "Temporal size for VAE tiled decode"}),
                "vae_temporal_overlap": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1, "tooltip": "Temporal overlap for VAE tiled decode"}),
            },
            "optional": {
                "image_embeds": ("CLIP_VISION_OUTPUT", ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Variable ratio image upscaling using 50% overlapping tile-based FramePack single frame sampling"

    def process(self, model, vae, shift, positive, negative, input_image, latent_window_size, 
                use_teacache, teacache_rel_l1_thresh, steps, cfg, guidance_scale, seed, 
                sampler, gpu_memory_preservation, tile_size=512, tile_resized=1024, 
                vae_tile_size=256, vae_overlap=64, vae_temporal_size=64, 
                vae_temporal_overlap=8, image_embeds=None, denoise_strength=1.0):
        
        upscale_factor = float(tile_resized) / float(tile_size)
        self._log_process_start(upscale_factor, vae_tile_size, vae_overlap, 
                               vae_temporal_size, vae_temporal_overlap, seed)
        
        # デバイス設定とモデル準備
        transformer, base_dtype, device, offload_device = self._setup_model(model)
        
        # 1. 画像を50%オーバーラップでタイルに分割
        tile_processor = TileProcessor(tile_size, tile_resized)
        tiles, metadata = tile_processor.split_image_to_tiles(input_image)
        
        # 2. テキストエンベッディング準備
        text_embeddings = self._prepare_text_embeddings(
            positive, negative, cfg, device, base_dtype
        )
        
        # 3. 画像エンベッディング準備
        image_encoder_embeddings = self._prepare_image_embeddings(
            image_embeds, device, base_dtype
        )
        
        # 4. ComfyUIセットアップ
        callback = self._setup_comfy_ui(transformer, device, steps, base_dtype)
        
        # 5. モデル準備
        self._prepare_model(transformer, device, gpu_memory_preservation, 
                           use_teacache, steps, teacache_rel_l1_thresh)
        
        # 6. 各タイルを処理
        processed_tiles = self._process_all_tiles(
            tiles, vae, transformer, sampler, denoise_strength, latent_window_size,
            cfg, guidance_scale, shift, steps, seed, device, base_dtype,
            text_embeddings, image_encoder_embeddings, callback,
            vae_tile_size, vae_overlap, vae_temporal_size, vae_temporal_overlap
        )
        
        # 7. 50%オーバーラップブレンドで最終画像を再構成
        reconstructor = ImageReconstructor(metadata)
        final_image = reconstructor.reconstruct_image(processed_tiles)
        
        # クリーンアップ
        self._cleanup(transformer, offload_device)
        
        print(f"=== 50%オーバーラップ {upscale_factor:.2f}xアップスケール完了: {final_image.shape} ===")
        return (final_image,)
    
    def _log_process_start(self, upscale_factor: float, vae_tile_size: int, 
                          vae_overlap: int, vae_temporal_size: int, 
                          vae_temporal_overlap: int, seed: int):
        """処理開始のログ出力"""
        print(f"=== FramePack 50%オーバーラップ Resize Sampler 開始 ({upscale_factor:.2f}x) ===")
        print(f"VAE Tile設定: size={vae_tile_size}, overlap={vae_overlap}, "
              f"temporal_size={vae_temporal_size}, temporal_overlap={vae_temporal_overlap}")
        print(f"使用シード: {seed} (全タイル共通)")
        print(f"オーバーラップ比率: {OVERLAP_RATIO*100:.0f}%")
    
    def _setup_model(self, model: dict) -> tuple:
        """モデルとデバイスのセットアップ"""
        transformer = model["transformer"]
        base_dtype = model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        
        return transformer, base_dtype, device, offload_device
    
    def _prepare_text_embeddings(self, positive, negative, cfg: float, 
                                device, base_dtype) -> dict:
        """テキストエンベッディングを準備"""
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
        
        return {
            "llama_vec": llama_vec,
            "llama_attention_mask": llama_attention_mask,
            "clip_l_pooler": clip_l_pooler,
            "llama_vec_n": llama_vec_n,
            "llama_attention_mask_n": llama_attention_mask_n,
            "clip_l_pooler_n": clip_l_pooler_n
        }
    
    def _prepare_image_embeddings(self, image_embeds, device, base_dtype):
        """画像エンベッディングを準備"""
        if image_embeds is not None:
            return image_embeds["last_hidden_state"].to(device, base_dtype)
        return None
    
    def _setup_comfy_ui(self, transformer, device, steps, base_dtype):
        """ComfyUIセットアップ"""
        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        return prepare_callback(patcher, steps)
    
    def _prepare_model(self, transformer, device, gpu_memory_preservation: float,
                      use_teacache: bool, steps: int, teacache_rel_l1_thresh: float):
        """モデルの準備（GPU移動とTeaCache設定）"""
        # モデルをGPUに移動
        move_model_to_device_with_memory_preservation(
            transformer, target_device=device, 
            preserved_memory_gb=gpu_memory_preservation
        )

        # TeaCacheの設定
        if use_teacache:
            transformer.initialize_teacache(
                enable_teacache=True, num_steps=steps, 
                rel_l1_thresh=teacache_rel_l1_thresh
            )
        else:
            transformer.initialize_teacache(enable_teacache=False)
    
    def _process_all_tiles(self, tiles: list, vae, transformer, sampler: str,
                          denoise_strength: float, latent_window_size: int,
                          cfg: float, guidance_scale: float, shift: float,
                          steps: int, seed: int, device, base_dtype,
                          text_embeddings: dict, image_encoder_embeddings,
                          callback, vae_tile_size: int, vae_overlap: int,
                          vae_temporal_size: int, vae_temporal_overlap: int) -> list:
        """全てのタイルを処理"""
        processed_tiles = []
        
        print(f"=== {len(tiles)}個のタイルを順次処理（50%オーバーラップ、各タイルで同一シード使用） ===")
        
        for i, tile_tensor in enumerate(tqdm(tiles, desc="オーバーラップタイル処理")):
            print(f"\n--- タイル {i+1}/{len(tiles)} 処理中（シード: {seed}） ---")
            
            # 各タイルで同じシードを使用するためにジェネレータをリセット
            rnd = torch.Generator("cpu").manual_seed(seed)
            
            processed_tile = self._process_single_tile(
                tile_tensor, vae, transformer, sampler, denoise_strength,
                latent_window_size, cfg, guidance_scale, shift, steps, rnd,
                device, base_dtype, text_embeddings, image_encoder_embeddings,
                callback, vae_tile_size, vae_overlap, vae_temporal_size, vae_temporal_overlap
            )
            
            processed_tiles.append(processed_tile)
            print(f"オーバーラップタイル {i+1} 完了: {processed_tile.shape}")
        
        return processed_tiles
    
    def _process_single_tile(self, tile_tensor: torch.Tensor, vae, transformer,
                           sampler: str, denoise_strength: float, latent_window_size: int,
                           cfg: float, guidance_scale: float, shift: float,
                           steps: int, rnd: torch.Generator, device, base_dtype,
                           text_embeddings: dict, image_encoder_embeddings,
                           callback, vae_tile_size: int, vae_overlap: int,
                           vae_temporal_size: int, vae_temporal_overlap: int) -> torch.Tensor:
        """単一タイルを処理"""
        tile_tensor = tile_tensor.to(device)
        
        with torch.autocast(device_type=mm.get_autocast_device(device), 
                          dtype=base_dtype, enabled=True):
            # VAEエンコード
            start_latent = self._encode_tile(vae, tile_tensor)
            
            # FramePack推論のためのインデックス設定
            latent_indices, clean_latents, clean_latent_indices = self._setup_latent_indices(
                start_latent, latent_window_size
            )
            
            # FramePack推論実行
            generated_latents = self._run_framepack_inference(
                transformer, sampler, denoise_strength, start_latent.shape,
                cfg, guidance_scale, shift, steps, rnd, device, base_dtype,
                text_embeddings, image_encoder_embeddings, latent_indices,
                clean_latents, clean_latent_indices, callback
            )
            
            # VAEタイル化デコード
            decoded_tile = self._decode_tile(
                vae, generated_latents, vae_tile_size, vae_overlap,
                vae_temporal_size, vae_temporal_overlap
            )
            
            # 時間次元を除去して画像形式に変換 [B, T, H, W, C] -> [B, H, W, C]
            if len(decoded_tile.shape) == 5:
                decoded_tile = decoded_tile.squeeze(1)
            
            return decoded_tile
    
    def _encode_tile(self, vae, tile_tensor: torch.Tensor) -> torch.Tensor:
        """タイルをVAEエンコード"""
        start_latent = vae.encode(tile_tensor)
        start_latent = start_latent * VAE_SCALING_FACTOR
        
        print(f"エンコード後latent: {start_latent.shape}")
        
        # latentを5次元に変換 [B, C, T, H, W]
        if len(start_latent.shape) == 4:
            start_latent = start_latent.unsqueeze(2)  # T次元を追加
        
        return start_latent
    
    def _setup_latent_indices(self, start_latent: torch.Tensor, 
                            latent_window_size: int) -> tuple:
        """FramePack推論のためのインデックス設定"""
        B, C, T, H, W = start_latent.shape
        
        # 1フレーム推論のためのインデックス設定
        sample_num_frames = 1
        latent_padding = 0
        latent_padding_size = latent_padding * latent_window_size
        
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        
        if latent_padding_size == 0:
            clean_latent_indices_pre = indices[:, 0:1]
            latent_indices = indices[:, 1:1+latent_window_size]
            clean_latent_indices_post = indices[:, 1+latent_window_size:2+latent_window_size]
            clean_latent_2x_indices = indices[:, 2+latent_window_size:4+latent_window_size]
            clean_latent_4x_indices = indices[:, 4+latent_window_size:20+latent_window_size]
            blank_indices = torch.empty((1, 0), dtype=torch.long)
        else:
            split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(split_sizes, dim=1)

        # 1フレーム推論設定
        all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
        latent_indices = all_indices[:, -1:]
        
        clean_latents_pre = start_latent.to(torch.float32).cpu()
        clean_latents_post = torch.zeros_like(clean_latents_pre)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        
        clean_latent_indices = torch.tensor([[0]], dtype=clean_latent_indices_pre.dtype, device=clean_latent_indices_pre.device)
        clean_latents = clean_latents[:, :, :1, :, :]
        
        return latent_indices, clean_latents, clean_latent_indices
    
    def _run_framepack_inference(self, transformer, sampler: str, denoise_strength: float,
                               latent_shape: tuple, cfg: float, guidance_scale: float,
                               shift: float, steps: int, rnd: torch.Generator,
                               device, base_dtype, text_embeddings: dict,
                               image_encoder_embeddings, latent_indices: torch.Tensor,
                               clean_latents: torch.Tensor, clean_latent_indices: torch.Tensor,
                               callback) -> torch.Tensor:
        """FramePack推論を実行"""
        B, C, T, H, W = latent_shape
        sample_num_frames = 1
        
        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler=sampler,
            initial_latent=None,
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
            prompt_embeds=text_embeddings["llama_vec"],
            prompt_embeds_mask=text_embeddings["llama_attention_mask"],
            prompt_poolers=text_embeddings["clip_l_pooler"],
            negative_prompt_embeds=text_embeddings["llama_vec_n"],
            negative_prompt_embeds_mask=text_embeddings["llama_attention_mask_n"],
            negative_prompt_poolers=text_embeddings["clip_l_pooler_n"],
            device=device,
            dtype=base_dtype,
            image_embeddings=image_encoder_embeddings,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=None,  # 2x, 4x latentsを無効化
            clean_latent_2x_indices=None,
            clean_latents_4x=None,
            clean_latent_4x_indices=None,
            callback=callback,
        )
        
        return generated_latents
    
    def _decode_tile(self, vae, generated_latents: torch.Tensor, vae_tile_size: int,
                    vae_overlap: int, vae_temporal_size: int, vae_temporal_overlap: int) -> torch.Tensor:
        """タイルをVAEデコード"""
        generated_latents = generated_latents / VAE_SCALING_FACTOR
        print(f"デコード前latent: {generated_latents.shape}")
        
        decoded_tile = VAEProcessor.tiled_decode(
            vae, generated_latents, 
            tile_size=vae_tile_size, 
            overlap=vae_overlap,
            temporal_size=vae_temporal_size,
            temporal_overlap=vae_temporal_overlap
        )
        
        return decoded_tile
    
    def _cleanup(self, transformer, offload_device):
        """リソースのクリーンアップ"""
        transformer.to(offload_device)
        mm.soft_empty_cache()


# ノードクラスマッピング
NODE_CLASS_MAPPINGS = {
    "FramePackSingleFrameResizeSampler": FramePackSingleFrameResizeSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePackSingleFrameResizeSampler": "FramePack Single Frame Resize Sampler (50% Overlap)", 
}