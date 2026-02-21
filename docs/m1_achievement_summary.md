# M1 達成までの実施サマリ

## 1. スコープ
本書は、M1（パイプライン健全性）達成までに実施した作業をまとめる。
- seq連続性
- 停止性（SIGINTでの終了）
- データ流通の基本健全性（drop/backpressure異常の抑制）

## 2. 実施内容

### 2.1 連続性・停止性の基盤整備
- FFTBlock 以降のシーケンス追跡を追加し、段間での連番破綻を可視化。
- FFTBlock 側でシーケンスの扱いを整理し、Frame合成時の不整合を低減。
- SeqTrace/SeqCheck系ログを追加し、`mismatch` / `seqprobe` を定量確認可能にした。

### 2.2 モニタ系の副作用低減
- `SignalStatsMonitor` の運用見直し（本線への負荷抑制）
- poolsize 見直し（評価時に `poolsize=32` を使用）

### 2.3 実行・評価の自動化
- M1評価用スクリプトを整備し、同条件複数runで再現性を確認。
- 代表結果:
  - `MILESTONE-1: PASS`
  - `shutdown=1, mismatch=0, seqprobe=0, sink_fail=0`

## 3. 追加した主な評価スクリプト
- `scripts/run_milestone1.sh`
- `scripts/check_milestone1.sh`

## 4. 成果
- M1: 達成（100%）
- パイプライン健全性の判定を自動化し、再現性確認まで完了。
