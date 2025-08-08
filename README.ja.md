# ComfyUI-FramePackWrapper_PlusOne

[ComfyUI-FramePackWrapper_PlusOne](https://github.com/tori29umai0123/ComfyUI-FramePackWrapper_PlusOne) は、[ComfyUI-FramePackWrapper](https://github.com/kijai/ComfyUI-FramePackWrapper)および[ComfyUI-FramePackWrapper_Plus](https://github.com/ShmuelRonen/ComfyUI-FramePackWrapper_Plus)から派生した、FramePackの1フレーム推論ノード(kisekaeichi対応)を含むフォークです。

本リポジトリは、 @tori29umai0123 氏の[依頼を受けて](https://x.com/tori29umai/status/1928692381735432320)公開用にフォークしました。

## 機能

- **1フレーム推論**: 基本的な1フレーム推論および、kisekaeichi方式に対応しています。技術的詳細は[musubi-tunerのドキュメント](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md)を参照してください。
- **F1サンプラー対応**: より高品質で時間的一貫性の高い動画生成のための改良されたF1方式を採用
- **LoRA統合**: 適切な重み付けと融合オプションを備えたHunyuanVideo LoRAの完全サポート
- **タイムスタンプ付きプロンプト**: 特定のタイムスタンプでプロンプトを変更できる動的動画の作成
- **柔軟な入力オプション**: 参照画像と空の潜在空間の両方で完全なクリエイティブコントロールが可能
- **解像度制御**: 最適な動画サイズのための自動バケット検出
- **ブレンド制御**: タイムスタンプ間の異なるプロンプトのスムーズな遷移

### 未対応機能

- 1フレーム推論のうち、f-mc (one frame multi-control) 方式は未対応です。

## インストール

1. このリポジトリをComfyUIのcustom_nodesフォルダにクローンします:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xhiroga/ComfyUI-FramePackWrapper_PlusOne.git
```

2. 必要な依存パッケージをインストールします:
```bash
pip install -r requirements.txt
```

3. 必要なモデルファイルをダウンロードして、modelsフォルダに配置します:
- FramePackI2V_HY: [HuggingFace Link](https://huggingface.co/lllyasviel/FramePackI2V_HY)
- FramePack_F1_I2V_HY: [HuggingFace Link](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503)

## モデルファイル

### メインモデルオプション
- [FramePackI2V_HY_fp8_e4m3fn.safetensors](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_fp8_e4m3fn.safetensors) - Optimized fp8 version (smaller file size)
- [FramePackI2V_HY_bf16.safetensors](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_bf16.safetensors) - BF16 version (better quality)

### 必要なコンポーネント
- **CLIP Vision**: [sigclip_vision_384](https://huggingface.co/Comfy-Org/sigclip_vision_384/tree/main)
- **Text Encoder and VAE**: [HunyuanVideo_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files)

## 使用方法

[example_workflows](./example_workflows)を参照ください。

| [1-Frame](./example_workflows/Oneframe.json) / [LoRA @tori29umai](https://huggingface.co/tori29umai/FramePack_LoRA/blob/main/Apose_V7_dim4.safetensors) | [1-Frame](./example_workflows/Oneframe.json) / [LoRA @kohya-ss](https://huggingface.co/kohya-ss/misc-models/blob/main/fp-1f-chibi-1024.safetensors) | [Kisekaeichi](./example_workflows/Oneframe_kisekaeichi.json) / [LoRA @tori29umai](https://huggingface.co/tori29umai/FramePack_LoRA/blob/main/body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors) |
| --- | --- | --- |
| ![kisekaeichi](./images/basic-apose.png) | ![chibi](./images/basic-chibi.png) | ![body2img](./images/kisekaeichi-body2img.png) |

## ライセンス

[MIT License](LICENSE)

## 更新履歴

### v2.0.0 - musubi-tuner完全互換対応 (2025-01-XX)

複数リファレンス画像使用時の推論結果の一貫性を改善するため、**musubi-tuner仕様との完全互換性**を実現しました。

#### 主要な変更点

**1. エンベッディング統合方式の改善**
- ❌ 従来：重み付き平均による統合処理（入力画像70%、参照画像30%）
- ✅ **新版：** musubi-tuner互換の処理方式（最初の参照画像エンベッディングを使用）

**2. Latent結合構造の統一**
- ❌ 従来：入力画像と参照画像を分離管理後に結合
- ✅ **新版：** musubi-tuner仕様のcontrol_latents直接結合
  ```python
  control_latents = [入力画像, 参照画像1, 参照画像2, ..., ゼロlatent]
  clean_latents = torch.cat(control_latents, dim=2)
  ```

**3. マスク適用タイミングの最適化**
- ❌ 従来：latent結合前の個別適用
- ✅ **新版：** clean_latents生成後にマスク適用（musubi-tuner仕様）

**4. インデックス設定の動的処理**
- ❌ 従来：固定的なclean_latent_indices設定
- ✅ **新版：** control_indicesパラメータの動的適用
  ```python
  # control_index="0;7;8;9;10" → clean_latent_indices = [0, 7, 8, 9, 10]
  while i < len(control_indices_list) and i < clean_latent_indices.shape[1]:
      clean_latent_indices[:, i] = control_indices_list[i]
  ```

**5. latent_indicesの初期化改善**
- ❌ 従来：ComfyUI独自の初期化方式
- ✅ **新版：** musubi-tuner仕様の初期化
  ```python
  latent_indices = torch.zeros((1, 1), dtype=torch.int64)
  latent_indices[:, 0] = latent_window_size  # デフォルト値
  latent_indices[:, 0] = target_index        # パラメータ適用
  ```

#### 期待される効果

- **推論結果の一貫性向上**: 同じリファレンス画像・同じパラメータでmusubi-tunerと完全に同じ結果を生成
- **複数リファレンス処理の安定化**: より正確なインデックス管理による安定した品質
- **パラメータ互換性**: musubi-tunerのcontrol_indexとtarget_indexパラメータが正しく動作

#### 技術的詳細

この更新により、以下の処理フローがmusubi-tunerと完全に一致します：

1. **制御画像処理**: `--control_image_path`で指定された複数画像の順次処理
2. **インデックス管理**: `--one_frame_inference="control_index=0;7;8;9;10,target_index=5"`の動的適用
3. **エンベッディング処理**: セクション毎の個別処理を模擬した実装
4. **マスク適用**: clean_latents構築後の統一的なマスク処理

## クレジット

- [FramePack](https://github.com/lllyasviel/FramePack): @lllyasviel 氏によるオリジナルの実装です。
- [ComfyUI-FramePackWrapper](https://github.com/kijai/ComfyUI-FramePackWrapper): @kijai 氏によるオリジナルの実装です。
- [ComfyUI-FramePackWrapper_Plus](https://github.com/ShmuelRonen/ComfyUI-FramePackWrapper_Plus): @ShmuelRonen 氏によるF1対応のフォークです。
- [ComfyUI-FramePackWrapper_PlusOne](https://github.com/tori29umai0123/ComfyUI-FramePackWrapper_PlusOne): @tori29umai0123 氏による1フレーム推論対応のフォークです。
- [musubi-tuner](https://github.com/kohya-ss/musubi-tuner): @kohya-ss 氏による高品質なFramePack学習・推論ライブラリ
