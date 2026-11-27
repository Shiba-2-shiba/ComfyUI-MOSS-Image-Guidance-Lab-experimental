# ComfyUI MOSS-Image Guidance Lab

ComfyUIネイティブのMOSS-Imageで、静止画のテキスト生成を比較する実験用ノードです。**SCGとLEAP** をそれぞれ独立に比較できます。画像編集、動画、部分デノイズ、他モデル、SCGとLEAPの併用、AFM・SLGなどとの組み合わせはこの版の検証対象外です。

## インストール

このディレクトリを `ComfyUI/custom_nodes/ComfyUI-MOSS-Image-Guidance-Lab` に置き、ComfyUIを再起動してください。ComfyUI本体やモデルは変更せず、ノード専用の追加パッケージはありません。ネイティブMOSS-Imageが動作する環境が必要です。実装の参照版はComfyUI v0.46.0です。使用中の環境とモデルは比較前に固定してください。

旧版がある場合は上書きではなく、旧フォルダを `custom_nodes` の外へ移動してから新フォルダを置いてください。旧版と新版が同時に読み込まれると、登録ノードが混在することがあります。[公式MOSS-Imageセットアップ](https://docs.comfy.org/tutorials/image/moss-image/moss-image)

### 比較時の注意

- ネイティブのサンプリング設定を変更しません。標準の `ModelSamplingSD3` でネイティブMOSSのshiftを調整しないでください。明示調整は `ModelSamplingAuraFlow` の `flow` モードで行い、対照も同じ値にしてください。
- SCGの配布グラフでは、`Load LoRA (Model)` → `MOSS Image Block Tune Preset`（`static`）の出力をガイダーとスケジューラーで共有します。BlockTuneには `AFM_for_MossImage_experimental` が必要です。素のモデルで比較する場合は、モデルローダーの出力をこの2か所に直接つないでください。
- 画質の評価は破綻、指示への一致、細部・質感、色を同じシードごとに確認します。再計時するときはComfyUIの実行キャッシュによる省略を除外してください。

## SCG・LEAPを試す

- **SCG**: `workflows/moss_scg_ab.json`。モデルを選び、プロンプトを入力して実行すると、同一シードの通常EulerとSCGを1枚ずつ保存します。SCGは画像入力なしで、MOSS用VAEの潜在空間で調整した論文の式 (7) を使います。任意の参照画像で比較したい場合は、`MOSS-Image SCG Profile` の出力を `SCG Euler` の `profile` に接続します。
- **LEAP**: `workflows/moss_leap_compare.json`。校正用と評価用のプロンプトを分け、Euler-25、Euler-15、LEAP（25更新・15回のguided評価）を比較します。初回は校正が走ります。
- [実機での手順と計測値の読み方](docs/scg-leap-validation-ja.md) · [実装計画書](docs/superpowers/plans/2026-11-23-moss-scg-leap.md)

SCGの実測ログは `MOSS-Image SCG Euler` の `debug_log` で有効にできます。`jsonl` は各ステップの補正量と周波数別の計測値、`jsonl+latents` は軌道比較用の潜在も保存します。[記録とEuler対照の比較手順](docs/scg-leap-validation-ja.md#scgのデバッグログ)

同じ補正量で帯域内の振幅配分を変える実験には、同ノードの `within_bin_mix` と `budget_trace` を使います。[現行SCGとの比較手順](docs/scg-leap-validation-ja.md#帯域内の振幅配分を比較する)

周波数を測る座標は `spectrum_space` で選べます。`packed_measured` と `spatial_measured` は同じ写真の実測スペクトルを使い、後者はFLUX.2の2×2パックを空間順に戻して補正します。[同じ総補正量で座標を比較する手順](docs/scg-leap-validation-ja.md#fft座標を比較する)

## 出典・検証範囲

[SCG](https://arxiv.org/abs/2610.02753) と [LEAP](https://arxiv.org/abs/2610.25918) の論文に基づく独立実装です。著者の別モデル用コード・重み・校正係数は同梱しません。公式データセットでの再現、MOSS-Imageの画質向上、本番GPU性能を確認済みとはしていません。

開発時はPyTorch、tqdm、pytestのある環境で `python -m pytest -q` を実行します。

ComfyUIの依存が入っている開発環境で、ComfyUIのルートを `PYTHONPATH` に含め、`python tests/native_workflow_schema_smoke.py`、`python tests/native_scg_smoke.py`、`python tests/native_leap_smoke.py` を実行すると、実際のComfyUIのV3ノード登録と配布グラフのソケット、小型のランダム初期化MOSSでのSCG/Euler動作、LEAPの実行キャッシュを検証できます。実機の生成画像や速度を再現するテストではありません。
