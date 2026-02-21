# M2 達成までの実施サマリ

## 1. スコープ
本書は、M2（FrameSync安定化）達成までに実施した作業をまとめる。
- unlock制御の安定化
- 低SNRストレス評価
- 通常SNRでの運用点確定

## 2. 実施内容

### 2.1 FrameSyncの可視化と制御追加
- `ISDBTFrameSync` に unlock抑制ガード（ref hold系条件下）を追加。
- unlock境界探索のためのCLIパラメータを追加:
  - `--framesync-unlock-threshold`
  - `--framesync-unlock-confirm`
- 周期ログで `unlock_suppressed` を可視化。

### 2.2 低SNRストレス環境の整備
- `AWGNInjector` ブロックを実装し、ソフト的にSNR劣化を注入可能化。
- `isdbt_demod.jl` に AWGN挿入オプションを追加。
- 初期に発生した AWGNフレームサイズ不一致（`FFTBlock total_samples=0`）を修正し、有効評価状態に復帰。

### 2.3 unlock境界探索
- `unlock_confirm=20` で `unlock_th=0.36/0.38/0.40` を比較。
- `0.40` 付近で unlock発生、`0.36/0.38` は維持（境界の再現性を確認）。
- `unlock_confirm=12` へ下げると `0.36` でも unlock/relockが増加し、チャタリング増大を確認。

### 2.4 通常SNRでの運用点確定
運用点候補を通常SNR (`AWGN 12/6/0 dB`) で再検証。
- `0.36/20`: 12dB・6dBでunlock多発（FAIL）
- `0.30/20`: 12dBでunlock残存（FAIL）
- `0.25/20`: 12/6/0dBすべて `unlock=0`（PASS）

確定値:
- `framesync_unlock_threshold = 0.25`
- `framesync_unlock_confirm = 20`

## 3. デフォルト設定への反映
- `examples/isdbt_demod.jl` の既定値を `unlock_th=0.25` に更新。
- これにより CLI引数未指定でも、M2で確定した運用点が適用される。

## 4. 追加した主な評価スクリプト
- `scripts/run_framesync_lowsnr_sweep.sh`
- `scripts/run_framesync_unlockth_sweep.sh`
- `scripts/run_framesync_unlockth_boundary.sh`
- `scripts/run_framesync_operatingpoint_validate.sh`
- `scripts/check_framesync_operatingpoint.sh`

## 5. 成果
- M2: 達成（100%）
- FrameSync運用点をデータに基づいて固定し、通常SNRで再検証を完了。
