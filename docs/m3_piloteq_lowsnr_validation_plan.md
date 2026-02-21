# M3-1 PilotEQ 低SNR動作検証計画

## 1. 目的
- 低SNR環境で `PilotEQ` の補正係数が時系列で安定する設定を決定する。
- `FrameSync` 影響を混在させないため、FrameSync運用点を固定した上で評価する。

## 2. 前提条件
- 固定運用点:
  - `framesync_unlock_threshold = 0.25`
  - `framesync_unlock_confirm = 20`
- ターゲット周波数: `515.142857 MHz`
- 入力: `ip:192.168.10.90`
- 低SNRは `AWGNInjector` で再現する。

## 3. 試験マトリクス
- SNR: `6 dB`, `0 dB`, `-2 dB`（必要なら `-4 dB` を追加）
- `pilot_temporal_alpha`: `0.1`, `0.2`, `0.3`
- 各条件の実行時間: `300 s`
- 実行本数: `3 (SNR) x 3 (alpha) = 9 run`

## 4. 観測項目
- 同期安定性:
  - `unlock`, `forced_resync`, `outlier_resync`
- PilotEQ下流の追従状態:
  - `PhaseSlope gate/updated`
  - `CPE conf/gate/updated`
- 品質指標:
  - `EVM`（有効化時は最重要）
- パイプライン健全性:
  - `seq mismatch/probe`
  - `FFTBlock sink_fail`
  - `Shutdown complete`

## 5. 合格判定（暫定）
- 必須条件:
  - `unlock=0`
  - `forced_resync=0`
  - `outlier_resync=0`
  - `sink_fail=0`
- 品質条件:
  - EVM平均が最良、かつ時間変動（ばらつき）が小さい設定を採用。
- 追従条件:
  - `updated` がゼロ固定でも乱発でもない（過抑制・過敏を回避）。

## 6. 実行順序
1. Baseline取得（`alpha=0.2`, `SNR=0 dB`, 300 s）
2. alpha sweep（各SNRで `0.1/0.2/0.3`）
3. ログ比較で候補alphaを選定
4. 候補alphaで確認run（`600 s` 推奨）

## 7. 実行コマンド雛形
```bash
timeout -s INT 300 julia --project=. ./examples/isdbt_demod.jl \
  -c 515.142857M \
  -i ip:192.168.10.90 \
  --diag \
  --no-const \
  --seq-trace \
  --seq-trace-log-interval 200 \
  --pilot-temporal-alpha 0.2 \
  --awgn-snr-db 0 \
  --awgn-log-interval 10
```

## 8. 出力物
- 条件別ログ（SNR/alpha）
- 条件比較表（lock/unlock, resync, EVM, update activity）
- 採用alphaと根拠
