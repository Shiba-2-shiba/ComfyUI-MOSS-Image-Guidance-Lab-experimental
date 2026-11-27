# MOSS-Image SCG / LEAP 実装計画書

**現行配布版の対象**: SCG と LEAP。ネイティブ MOSS-Image の静止画 T2I のみ。ComfyUI v0.46.0 を基準に、ComfyUI と PyTorch 以外の製品依存なし。本体は無改変。実機の最終確認はユーザー。ノードは別々に試し、併用、Edit、動画、マスク、部分デノイズ、ER-SDE への移植を含めない。論文に基づく独立実装。学習済み係数・重み・第三者コードを同梱しない。公開・push はこの工程に含めない。

## 比較の軸

- SCG: 同一 Euler の無補正と補正を同じチェックポイント、VAE、シード、プロンプト、寸法、CFG、sigma 列で比較。
- LEAP: Euler-25（同じ sigma 列）、Euler-15（同じ予定 guided 回数）、LEAP-25/15（25 更新中15 guided 評価）。シード/初期ノイズ共通。校正と比較のプロンプト・シードは分離。
- 1024×1024、CFG 2.5 を初回の提案値とし、比較画像の指示一致、文字、細部、色・粒状感、破綻を評価。速度・画質は実測前に保証しない。

## Task 1: SCG の数式と ComfyUI 接続

- [ ] SCG 専用モジュールと CPU テスト。ネイティブの同じ VAE で許諾済み参照画像の latent を作り、MOSS の process_latent_in を一度だけ通す。画像/latent サイズを校正時と照合する。参照は推論評価画像と独立。
- [ ] profile 入力ノード MossImageSCGProfile、SAMPLER 出力 MossImageSCGEuler。正規化 packed Flux2 の NCHW shape、VAE/モデルの同一性、FFT 規約などを検査可能な範囲で照合し、検査不能な条件は利用者の同一構成手順として明示する。VAE の tiled fallback は確認できない限り不明と記録。
- [ ] FP32 full FFT orthonormal、ビン内の平均 power、標準ノイズ期待 power=1。Euler 更新の直後、次の実 sigma で T=(1−sigma)^2 P_clean+(sigma · noise_scale)^2 を計算し、((T+eps)/(S+eps))^(strength/2) の有限クランプ済み振幅ゲインを使用。位相と DC、ビン内の相対振幅を保ち、空/零振幅ビンは描き足さない。論文の収束・画質を保証しない。経験的参照、振幅キャップ、窓、終端 sigma=0 の省略は実験版の選択として明記。
- [ ] strength=0 は完全にネイティブ Euler に委譲。縦横・odd/even・定数・単一モード・バッチ・有限値・次 sigma・no-op・実 forward 数のテスト。

## Task 2: LEAP の数式・校正・係数

- [ ] LEAP の独立モジュールで Eq. 25–31 / Algorithm 2 を実装。guided velocity=(x−denoised)/sigma。r=5 の基底は時系列順に古い値マイナス次の値を4本、最後に最新値。予測値を履歴に入れ、次回の真評価は予測で到達した状態から算出。25 更新では i=5,7,…,23 を予測し、終端前 i=24 は真評価。
- [ ] フル Euler-25 軌跡からタイムステップ別 FP64 総和 Gram/RHS と固定 ridge 1e-4 の明示校正。2/8/32 軌跡は試験規模の目安で品質保証ではない。推論時に暗黙再校正しない。漏洩しないよう校正と評価は別 seed/条件。係数は安全に再利用できる版管理付きデータ、または同一セッションへ厳しく束縛する。ファイルを使うならデータ形式のみで pickle 不使用。版、入力形状、実 sigma 列、CFG、モデル/量子化、sampling、shift/multiplier、ランタイム特性との照合と限界を明記。
- [ ] 独立 dense ridge との一致、分割統計、差分符号、整形、履歴、NFE、非有限・非互換・無効時 Euler をテスト。

## Task 3: ComfyUI LEAP と計時

- [ ] ComfyUI V3 の校正・Euler/LEAP SAMPLER アダプター。既存 native guider を経由し、実モデル型/予測形式、最終 sigma=0、マスク・競合フックをチェック。未校正や不一致は明示エラー。タイムステップ名目数を物理 forward と取り違えない。
- [ ] guided 呼出数、native diffusion module の物理 forward 数と各バッチサイズ、warm sampler 実時間を各試行にひも付ける。PyTorch CUDA/HIP は実デバイスの同期待ちを開始直前と停止直前にのみ入れる。GPU/CPU 転送を伴う診断はループ中に置かない。計時範囲を native サンプラー区間とキュー開始〜画像保存（別計測）で分離。校正時間は別。タイムスタンプだけでキャッシュヒットを実推論扱いしない。
- [ ] 反復はベンチ用 nonce を入力にし、seed・noise テンソルや RNG を変えず必要なサンプラーのみ無効化。キャッシュ命中で forward 0 の試行は不採用。CFG1/実運用CFG、意図的なキャッシュ再利用、CUDA/HIP境界、比較ごとの Sigma/回数を確認。
- [ ] SCG/LEAP の登録、ComfyUI HEAD でなく基準コミットでネイティブ境界を可能な限り確認。CPU の疑似モデルと実モデル重み・実機生成を報告上分ける。

## Task 4: ワークフロー・受入・配布

- [ ] SCG と LEAP を独立に試せる workflow/実行手順。チェックポイント/VAE の指定、参照/校正取得、同 seed、ファイル名、3 群の実時間と物理 forward の読み方、比較順、初回ウォームアップと無効試行を記載。列の意味と調整が必要になる条件を説明。
- [ ] 実装後、CPU pytest、compileall、ワークフロー JSON/接続検証、可能なら固定版 ComfyUI synthetic smoke を実行。重み・Windows/R9700・生成画像・現実の速度は未検証なら明記。コードと計画書をまとめて渡す。

## 原典
- SCG: https://arxiv.org/html/2610.02753v1
- LEAP: https://arxiv.org/html/2610.25918v1
- ComfyUI v0.46.0: https://github.com/Comfy-Org/ComfyUI/tree/a277b5bf213e8a44fa34817b7770b54f86ef4cea
