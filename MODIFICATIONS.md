# Modifications from upstream YaneuraOu

Upstream: https://github.com/yaneurao/YaneuraOu
Base commit: `cc73ac44ad1433463df73fc6c97c06fd27c5d266` (2026-05-09、`UPSTREAM.md` 参照)
License: GPLv3 (`Copying.txt`)

本フォークは上流 YaneuraOu を基点に、将棋NNUE研究プロジェクト **Kisou Engine (棋想)** の
配布・実験用に以下を改変・追加した (GPLv3 §5 の改変告知)。

## ★配布版 (Kisou Engine) の挙動に直接効く改変

配布バイナリはこの 2 点を含む。**上流の探索既定値とは異なる。**

### 1. 探索パラメータの再調整 (SPSA)

`source/engine/yaneuraou-engine/yaneuraou-search.cpp`:
razoring / futility / null move / LMR reduction / aspiration window / ProbCut /
SEE margin / singular extension / IIR の各定数 **32 個**を `TUNABLE_PARAM` マクロで定義し、
**SPSA (Fishtest 方式) による最適化結果を既定値として反映**した。

```
#if defined(ENABLE_SEARCH_TUNE)
    #define TUNABLE_PARAM(name, val) int name = val;      // USI オプションとして可変
#else
    #define TUNABLE_PARAM(name, val) constexpr int name = val;  // 焼き込み (配布版)
#endif
```

- `ENABLE_SEARCH_TUNE` 未定義 (= 配布ビルド) では `constexpr` のままなので、
  上流と同じくコンパイル時定数として畳み込まれる。**NPS への影響は無い** (実測確認済み)。
- `ENABLE_SEARCH_TUNE` 定義時のみ USI オプションとして露出し、Stockfish の `TUNE()` 機構で
  SPSA から駆動できる。`Tune::init(options)` は上流に既に在るものをそのまま使う。

### 2. NNUE アーキテクチャ `halfkp_768x2-16-32` の追加

配布版 (v1、2026-08-30 更新) の評価関数 (`eval/nn.bin`) が使うアーキテクチャ。
`source/eval/nnue/architectures/halfkp_768x2-16-32.h` (新規) と
`source/eval/nnue/nnue_architecture.h` / `source/Makefile` の分岐。
配布ビルドは L1 を int8 QB=128 で量子化した net を使うため
`EXTRA_CPPFLAGS=-DNNUE_L1_SCALE_BITS=7` を付けてビルドする
(net 側 description の `;L1QB=128/i8` タグと照合し、不一致は起動時に拒否)。
旧配布版 (2026-08-15 公開〜2026-08-30) は `halfkp_512x2-16-32.h` (q64、シフト 6) を使用。

## 配布・移植性

- `source/eval/nnue/evaluate_nnue.cpp` / `source/misc.cpp` / `source/misc.h`:
  評価ファイル (nn.bin) のロードを **wide(UTF-16) Windows API 化**
  (`Directory::ReadBinaryFolderRelativeFileW` = `GetModuleFileNameW` + `CreateFileW`)。
  ANSI コードページ (例 CP932) で表現できない文字を含むパス (韓国語/絵文字/混在スクリプト)
  でも評価関数を読めるようにする。**探索・評価の数値は不変** (ロード経路のみ変更)。
- `dispatcher/`: CPU の SIMD を判定し `engine\YaneuraOu-<ISA>.exe` を透過起動するランチャ
  (YaneuraOu 本体とは独立したプログラム。CreateProcessW で子を起こし stdio を継承 =
  USI プロトコルに介入しない)。
- `packaging/`: 配布パッケージ生成スクリプト。GPLv3 の「ビルド/インストールを制御する
  スクリプト」に相当。
- `source/tune.cpp`: Fishtest 用パラメータ CSV の出力先を `std::cout` → `std::cerr` に変更。
  stdout は USI プロトコルのストリームであり、非 USI 行を混ぜると GUI や対局ドライバの
  パーサが壊れるため。

## 実験用アーキ variant (コンパイルフラグ後ろ、配布ビルド非影響)

`source/eval/nnue/architectures/` に以下を追加し、`nnue_architecture.h` と `Makefile` に分岐を追加:

| ヘッダ | 用途 |
|---|---|
| `halfkp_512x2-8-64.h` | Suisho10 と同一アーキ (比較用) |
| `halfkp_512x2-32-32.h` | L2 幅の検証 |
| `halfkp_512x2-16-32-screlu.h` | FT 活性化を SCReLU 化 |
| `halfkp_512x2-16-32.h` | 旧配布アーキ (2026-08-30 の v1 更新まで) |
| `halfkp_768x2-8-32.h` | FT 幅 768 の検証 |
| `halfkp_768x2-16-64.h` | AobaNNUE 1.1 と同一アーキ (比較用) |
| `halfkp_1024x2-8-32.h` / `halfkp_1024x2-8-64.h` | FT 幅 1024 |
| `halfka_512x2-16-32.h` | HalfKA 入力 |
| `halfkp_256x2-32-32-screlu.h` / `-pairwise.h` | 256 系の活性化 variant |
| `sfnnwop-1536.h` | SFNN 型 (upstream 付属) |
| `SFNNwoPSQT_halfka2_1024-7-64-ls9.h` | SFNN HalfKA2 1024-7-64 9スタック (Suisho11Plus と同型)。<br>upstream の `nnue_arch_gen.py` による自動生成物。Phase B 用 |

- `source/eval/nnue/nnue_feature_transformer.h`: SCReLU (`NNUE_FT_SCRELU`) /
  pairwise (`NNUE_FT_PAIRWISE`) の FT 変換と hash マーカー (AVX2/scalar のみ)。
- `source/eval/nnue/evaluate_nnue.cpp`: SCReLU/pairwise ビルドで nn.bin の hash 不一致を
  hard error にする (異種ネット混載の拒否)。
- `source/misc.cpp`: `config_info()` に追加 arch 名。

## データ生成・検証・計測ツール (対局中は呼ばれない)

- `source/learn/filter_quiet.cpp` (新規) + `source/usi.cpp`: `filter_quiet` USI サブコマンド
  (静止局面フィルタ、データセット生成用。`USE_SFEN_PACKER` ガード)。
- `source/engine/yaneuraou-engine/yaneuraou-search.cpp`: `trace_eval()` が静的評価値を
  `eval = <値>` 形式で出力 (元は空スタブ)。`eval` USI コマンド実行時のみ動作、探索挙動は不変。
  学習器と推論の数値一致検証 (`tools/nnue_eval.py --verify`) に使う。
- `source/eval/nnue/evaluate_nnue.cpp`: テールゲイン単調変換 `g` (USI オプション
  `GTAIL_T` / `GTAIL_GAIN`)。**既定値 `GTAIL_GAIN=100` は恒等変換**なので、
  設定しない限り上流と同一の評価値を返す。
- eval 呼び出し統計の計測パッチ (`EVAL_LOG_PATH` 未設定なら完全無効)。
- `source/eval/nnue/evaluate_nnue.cpp`: **合成 small eval ゲートのシミュレータ**
  (`EvalGateSim`, report/45 Phase-0)。区間型の軽量評価で探索の境界判定を肩代わりさせたとき
  探索品質がどれだけ落ちるかを、実際の軽量ネットを作らずに測るための計測コード。
  `small = large + ノイズ(局面キーでシード)` を合成し、区間が窓境界 (alpha/beta) を
  またぐときだけ本物の評価値へ escalate する。USI オプション `GateE` (ノイズ標準偏差 cp、
  既定 0 = 無効) / `GateC` (区間半幅 = GateC/100 × GateE)。
  **`ENABLE_EVAL_GATE_SIM` を定義したビルドにのみ存在する。配布ビルドには含まれない。**
  ★実装上の注意: ゲート本体は `noinline`。inline 展開させると `evaluate()` から
  accumulator 更新までの codegen が変わり、**ゲート無効時 (`GateE=0`) でも**探索開始直後に
  アクセス違反で落ちる (2026-08-16 実測)。計測コードは呼び出し境界で切ること。
- `source/eval/nnue/nnue_feature_transformer.h` + `evaluate_nnue.cpp`:
  **FT メモリトラフィックの内訳カウンタ** (`ENABLE_FT_TRAFFIC_STAT`, task#45 / report/49)。
  accumulator の行読みが「差分更新」「差分連鎖の断絶による全再構築」「王移動による
  perspective 単位の再構築」のどれに使われているかを**行数で**数える。
  環境変数 `FT_TRAFFIC_STATS` にファイル名を指定するとプロセス終了時に追記する。
  **定義したビルドにのみ存在する。配布ビルドには含まれない。**
  配布ビルドとの同一性 (score/bestmove/nodes) を 8/8 局面で確認済み。

## 探索の改変 (対局挙動に影響しうるもの)

- `source/engine/yaneuraou-engine/yaneuraou-search.cpp`: **accumulator 差分連鎖の維持**
  (`ENABLE_ACC_CHAIN_FIX`, task#45 / report/49)。
  `search()` / `qsearch()` の王手局面 (`ss->inCheck`) は `evaluate()` を呼ばずに
  指し手ループへ飛ぶため、その局面の accumulator が未計算のまま子へ降りる。
  `UpdateAccumulatorIfPossible()` は **1 手前しか遡らない**ので、子は全再構築
  (両手番で ~76 行) に落ちる。ここで `Eval::evaluate_with_no_return()` を呼んで
  差分だけ進めておくと、子は差分 (~4 行) で済む。
  実測: 全再構築の回数 860,095 → 12,883 (−98.5%)、FT の総行読み −27.6%。
  **評価値を返さないので探索結果は変わらない** (同一局面・同一ノード数で
  score/bestmove/nodes が完全一致することを確認済み。transform 回数も 10,106,882 で一致)。
  ★ 2026-08-19 時点では**既定で無効**。NPS 改善が負荷下の測定では有意でなく
  (比 1.0231 / ノイズ床 ±2.78%)、アイドル再測の結果を見てから既定化を判断する。
- `source/history.h` + `source/engine/yaneuraou-engine/yaneuraou-search.cpp`:
  **material correction history** (`ENABLE_MATERIAL_CORRHIST`, task#51 / report/50)。
  Stockfish PR #5556 の将棋版。materialKey で index する補正履歴を
  correction_value / update_correction_history に追加。
  **定義したビルドにのみ存在する。既定無効。Elo 判定中 (採用が決まるまで配布ビルドに含まれない)。**
- `source/eval/nnue/features/threat.h/.cpp` (新規) + `architectures/halfkp_threat_512x2-16-32.h` (新規)
  + `nnue_architecture.h` + `Makefile`: **Threat 入力特徴** (task#52 / report/51)。
  駒の実利きが駒に当たっている関係 (玉除外、9クラス×敵味方、空盤幾何圧縮、216,720 次元) を
  HalfKP に連結した実験アーキ `YANEURAOU_ENGINE_NNUE_HALFKP_THREAT_512X2_16_32`。
  bullet-shogi 側 (`shogi_halfkp_threat.rs`) と同一 index 仕様で、テーブルの FNV-1a
  チェックサム (0x30f7eea2484893cd) を両実装に焼き込み、不一致なら起動時に即死する。
  差分更新は未実装 (kAnyPieceMoved で毎手全再構築 = 実験判定用)。**配布ビルドには含まれない。**

## SFNN + Threat 実験アーキ (halfka2t) の追加 (2026-08-22, task#54)

- `source/eval/nnue/features/threat_ka2.h` (新規、ヘッダのみ):
  `ThreatKa2` — Threat (threat.h) の HalfKA2 ペア用バリアント。index 計算・次元・
  trigger は Threat を継承し、kHashValue だけ 0xB52D879C に変更
  (bullet 側 composite 0x0b6b1eec = FEATURE_HASH_HALFKA2 ^ "THRT" に合わせた逆算値)。
- `source/eval/nnue/architectures/nnue_arch_gen.py`: 入力特徴 `halfka2t` を追加
  (`FeatureSet<ThreatKa2, HalfKA2<kFriend>>` — FeatureSet は Tail が offset 0 なので
  bullet レイアウト [KA2][Threat] と一致)。
  ビルド例: `YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_SFNNwoPSQT_halfka2t_1024-7-64-ls9`。
  **配布ビルドには含まれない** (研究判定用)。

## Threat 差分更新 (2026-08-22, task#37)

- `source/eval/nnue/features/threat.h`: kRefreshTrigger を kAnyPieceMoved → **kNone**
  (常に差分計算)。-DTHREAT_NAIVE_REBUILD で旧挙動。
- `source/eval/nnue/features/threat.cpp`: `AppendChangedIndices` 実装。
  影響 attacker (動いた駒/取られた駒/from・to に利く駒) の被害者集合を prev/now 占有で
  再列挙し対称差分。駒レベル差分は thread_local キャッシュで両視点共有。
  平均 6.78 行/手 (実測)。-DTHREAT_DIFF_STATS で診断カウンタ。
- `source/config.h`: threat edition で KEEP_LAST_MOVE を有効化 (FOR_TOURNAMENT の
  #undef より後で再定義)。
- **検証**: diff vs naive の固定ノード探索一致 24/24 (Windows clang) + 4/4 (Linux gcc)。
- NPS: plain-512 比 0.3988 (naive) → **0.5314** (diff)。残余は threat FT 行 (212MB 行列)
  のコールドフェッチが律速 (report/51 §7.4)。


## ThreatLite 特徴量 (2026-08-24, task#59)

- `source/eval/nnue/features/threat_lite.h/.cpp` (新規):
  `ThreatLite` — Threat の from/幾何を落とした縮約版
  (attacker_side, attacker_class, defender_side, defender_class, to) = 26,244 次元。
  bullet-shogi 側 `shogi_halfkp_threatlite.rs` と同一 index 仕様。
  ★count 意味論: 同一 (pair,to) への複数攻撃は同一 index を重複 push (特徴値=攻撃駒数)。
  当面はナイーブ全再構築 (kAnyPieceMoved)。差分化は Elo 保持率 A/B 通過後。
- `source/eval/nnue/architectures/halfkp_threatlite_512x2-16-32.h` (新規):
  FeatureSet<ThreatLite, HalfKP<kFriend>> の 512x2-16-32 型 (151,632 次元)。
- `source/eval/nnue/nnue_architecture.h` / `source/Makefile`:
  YANEURAOU_ENGINE_NNUE_HALFKP_THREATLITE_512X2_16_32 エディション追加。
- `source/eval/nnue/features/threat.h/.cpp`: KEEP_LAST_MOVE 無しビルドを
  自動で naive (kAnyPieceMoved) に落とすフォールバックを追加。
- ハッシュ検証: full-threat ネット読込で期待通り拒否、差分 0x00040611 =
  bullet 側タグ差 ("THRT"^"TLTE") と厳密一致 (= bullet export と相互整合)。

## 層別重みスケール + ClippedReLU 四捨五入オプション (2026-08-27, task#65)

- `source/eval/nnue/layers/affine_transform.h` / `affine_transform_sparse_input.h`:
  テンプレート引数 `WeightScaleBits` (既定 kWeightScaleBits=6) を追加し `kWeightScaleBits` を公開。
- `source/eval/nnue/layers/clipped_relu.h`: 前段の `kWeightScaleBits` でシフト (層別)。
  `-DNNUE_ROUND_SHIFT` で床シフトを四捨五入に (検証用オプション、既定 off)。
- `architectures/halfkp_threat_512x2-16-32.h` / `halfkp_threatlite_512x2-16-32.h`:
  L1 のスケールを `NNUE_L1_SCALE_BITS` (既定 6) で切替可能に (`-DNNUE_L1_SCALE_BITS=7` = QB 128)。
- 背景: L1 (1024→16) の int8 QB=64 量子化で重みの 3〜4 割が 0 に丸められ、fp32 比 −66cp/std 70 の
  ズレが出ていた (report/51 §7.7.1)。QB=128 で残差 p50 74→20cp。
- ★ネットワークハッシュはスケールビットに依存しないため、q64/q128/int16 のファイル取り違えはハッシュでは検出されない。
  そのため `evaluate_nnue.cpp` に `QuantTag()` を追加 (2026-08-28): nn.bin の description 末尾の
  `;L1QB=<qb>/<i8|i16>` をビルドの `NNUE_L1_SCALE_BITS` / `NNUE_SFNN_L1_SCALE_BITS` / `NNUE_L1_INT16` と照合し、
  不一致なら FileMismatch で読み込みを拒否する。タグ無しファイルは従来量子化 (QB64/int8) とみなす
  (従来ビルドでは従来ファイルがそのまま読める)。タグの付与は shogi-nnue 側 `tools/nnue_tag.py`。
- SFNN 経路も層別化 (2026-08-27): `layers/clipped_relu_explicit.h` / `layers/sqr_clipped_relu.h` に
  `WeightScaleBits` テンプレート引数 (SqrClippedReLU の SIMD 後シフトは 2*bits-9 に一般化)。
  `architectures/nnue_arch_gen.py` と生成済み SFNN ヘッダ (halfka2/halfka2t) で fc_0 の活性を
  `NNUE_SFNN_L1_SCALE_BITS` (既定 6) で切替、fc_0 の shortcut 出力は `>> (bits-6)` でスケール差を吸収。
- `layers/affine_transform_sparse_input_i16.h` (新規, 2026-08-28, task#68): L1 の int16 重み版疎入力 affine。
  classic 3 ヘッダで `-DNNUE_L1_INT16` により選択 (WeightScaleBits ≤ 8)。ハッシュは int8 版と同一。

## threat FT 行の attacker-major 並び替え + 行 prefetch (2026-09-05, task#59)

- `source/eval/nnue/features/threat.h/.cpp`: `-DTHREAT_ATTACKER_MAJOR` で threat index を
  (as, ac) ブロック内 (from, ord) スラブ × 18 (ds, dc) の attacker-major 配置に変更。
  nn.bin は標準 (pair-major) のまま、FT ロード時に `Threat::PermuteRows` が行を置換
  (全単射検算付き)。置換済み配置の WriteParameters は封鎖 (他ビルドで読めないファイル防止)。
  検証: 同一 nn.bin で標準ビルドと eval 80 局面ビット一致。NPS は ×1.0038 n.s. = 採用見送り
  (局所性仮説の棄却データとして保存)。
- `source/eval/nnue/nnue_feature_transformer.h`: `-DFT_ROW_PREFETCH` で update_accumulator の
  差分行 (removed/added、両視点) の先頭+中間ラインを accumulate 前に一括プリフェッチ。
  意味論不変。

## ThreatLite の差分更新 (2026-09-05, task#59 ②)

- `source/eval/nnue/features/threat_diff.h` (新規): Threat の駒レベル差分 (対の増減) を
  共有型 ThreatPiecePair/ThreatPieceDiff + `threat_piece_diff(pos)` として公開。
- `source/eval/nnue/features/threat.cpp`: 上記へのリファクタ (挙動不変、キャッシュは従来どおり)。
- `source/eval/nnue/features/threat_lite.h/.cpp`: kRefreshTrigger を kNone (差分) に変更し、
  AppendChangedIndices を threat_piece_diff の lite index 写像で実装。count 意味論なので
  参照カウント不要 (対の ±1 = 行の ±1 加算)。KEEP_LAST_MOVE 無し / -DTHREAT_NAIVE_REBUILD は naive に落ちる。
- `source/config.h`: THREATLITE エディションでも KEEP_LAST_MOVE を定義。
- 検証: 探索一致 20 局面 × 200k ノード (bestmove/score/nodes/pv 一致)。
  NPS: vs naive x1.4863 / vs full(diff) x1.0681 (訓練同居 busy 0.3、idle 再測は後続)。

## ThreatDrop2 特徴 (2026-09-05, task#73)

- `source/eval/nnue/features/threat_drop2.h/.cpp` (新規): ThreatLite から攻撃駒種も落とした
  (attacker_side, defender_side, defender_class, to) 2,916 次元の特徴。count 意味論、ナイーブ全再構築
  (Elo 保持率 A/B 用)。`-DTHREAT_DROP2_DUMP` で active index を stderr に出す (bullet 側との照合用)。
- `source/eval/nnue/architectures/halfkp_threat_drop2_512x2-16-32.h` (新規)、`nnue_architecture.h`、`Makefile`:
  エディション YANEURAOU_ENGINE_NNUE_HALFKP_THREATDROP2_512X2_16_32 を追加。
- 検証: index 多重集合が bullet-shogi と 64 局面一致、整数忠実 numpy forward と eval がビット一致。

## ThreatEffect 特徴 (2026-09-06, task#73 v2)

- `source/eval/nnue/features/threat_effect.h/.cpp` (新規): (attacker_side, defender_side, defender_class, to,
  長い利き 0/1/2+, 短い利き 0/1/2+) 26,244 次元。攻撃側は玉を含み、長い利き = 香/角/飛の射程 + 馬斜め + 龍縦横
  (LONG_EFFECT_LIBRARY の board_effect / long_effect と同定義 → 将来は列挙なしで差分計算可)。
  判定用はナイーブ全再構築。`-DTHREAT_EFFECT_DUMP` で index を stderr へ。
- `source/eval/nnue/architectures/halfkp_threat_effect_512x2-16-32.h` (新規)、`nnue_architecture.h`、`Makefile`:
  エディション YANEURAOU_ENGINE_NNUE_HALFKP_THREATEFFECT_512X2_16_32。
- 検証: bullet-shogi `ShogiHalfKPThreatEffect` と index 多重集合が 64 局面 × 両視点で一致。

## ThreatEffect の利き盤ベース差分更新 (2026-09-07, task#73 v2)

- `source/eval/nnue/features/threat_effect.h/.cpp`: 既定を `THREAT_EFFECT_DIFF` に変更。
  バケットを LONG_EFFECT_LIBRARY の `board_effect` (利き数) と `long_effect` (長い利き方向の popcount)
  から**列挙なし**で引き、差分更新は直前局面の利き盤との比較 (動いた駒/取られた駒 + 利き状態が変わった升の駒)。
  kRefreshTrigger は kNone (玉移動でも reset しない)。`-DTHREAT_NAIVE_REBUILD` で従来のナイーブ実装。
- `source/position.h`: StateInfo に `te_board_effect_prev[2]` / `te_long_effect_prev` を追加 (THREAT_EFFECT_DIFF 時)。
  `source/position.cpp` do_move(): 利き更新前の盤を新 StateInfo へ退避 (HalfKPE9 の Position 側 board_effect_prev と
  違い StateInfo に持つので undo/null move の順序に依存しない)。
- `source/config.h`: THREATEFFECT エディションで KEEP_LAST_MOVE / LONG_EFFECT_LIBRARY / THREAT_EFFECT_DIFF を定義。
  (LONG_EFFECT_LIBRARY により 1 手詰めルーチンが利き版に切り替わる。)

## SFNN halfka2te (HalfKA2 + ThreatEffect) edition (2026-09-07, task#73 王者移植)

- `source/eval/nnue/features/threat_effect_ka2.h` (新規): ThreatEffect の HalfKA2 ペア用バリアント (ハッシュのみ 0xB52093FA、
  BulletOu 側 composite 0x0B660A8A から逆算)。
- `source/eval/nnue/architectures/nnue_arch_gen.py`: 入力特徴 `halfka2te` を追加 (`FeatureSet<ThreatEffectKa2, HalfKA2>`)。
  edition `YANEURAOU_ENGINE_NNUE_SFNNwoPSQT_halfka2te_1024-7-64-ls9` で生成されるヘッダも同梱。
- `source/Makefile`: edition 名に `_halfka2te_` を含むとき `-DNNUE_SFNN_HALFKA2TE` を定義。
  `source/config.h`: それを受けて KEEP_LAST_MOVE / LONG_EFFECT_LIBRARY / THREAT_EFFECT_DIFF を有効化
  (SFNN でも利き盤ベースの差分更新を使う)。

## accumulator キャッシュ (Finny table) `ENABLE_ACC_CACHE` (2026-09-08, task#50)

- `source/eval/nnue/nnue_feature_transformer.h`: 玉移動トリガ (kFriendKingMoved) の accumulator slot を、視点 × 自玉升ごとの
  thread_local キャッシュ (その時の accumulator + ソート済 active index) との集合差分で作る。全再構築 (refresh) と玉移動 reset の
  両方に適用。特徴集合に依存せず index 列だけで動く (HalfKP / HalfKA2 共通)。ネット再読込は generation で無効化。
  整数加算の順序非依存性により結果はビット一致 (lite-diff / plain-512 で探索一致 20/20)。
  `-DENABLE_ACC_CACHE` で有効 (既定は無効)。FT_TRAFFIC_STAT に rows_cache / cache_hit を追加。
- 実測 (lite-diff + chainfix、20 局面 × 200k): 全再構築行 13.5M → 0.17M、キャッシュ差分 8.3 行/回、総行 −14.9%。
- `source/engine/yaneuraou-engine/yaneuraou-search.cpp`: qsearch TT ヒット経路の連鎖維持 (chainfix v2) は計測の結果
  不採用 (コメントで記録)。

## 定跡サブシステムを upstream (origin/master) から移植: `.ybb` 対応 (2026-09-10, task#81)

- `source/book/book.h` / `book.cpp` / `makebook.cpp` / `makebook2025.cpp` (+ `apery_book.*`, `policybook.*`): upstream master の版に置換。
  やねうら王バイナリ定跡DB (`.ybb`, "YANE-BINBOOK-V1") の読み込み (BookOnTheFly=true は index 二分探索、false は丸読み)、
  `BookFile=user_book1.db` で `.db` が無ければ `.ybb` に fallback、`makebook peta_shock` の `.ybb -> .ybb`、
  優先定跡 (`user_book1-000.db` 等) の複数保持。`makebook2015.cpp` (旧 makebook コマンド群) は upstream 同様に削除。
- `source/extra/sfen_packer.cpp`: `SfenPacker::pack_rawdata` / `unpack_rawdata`、`PackedSfen::flipped()` / `flip()` を追加
  (FlippedBook の packed sfen 直接 flip probe 用)。`USE_SFEN_PACKER` ガードはこのフォークの構成に合わせて維持。
  `source/position.h`: `PackedSfen` に `flip()` / `flipped()` を宣言。
- `source/usioption.h` / `usioption.cpp`: `OptionsMap::read_engine_option_profile()` (`engine_option_profile.txt` の `BOOK_OPTIONS=V2` で
  `BookEvalBlackDiff` / `BookEvalWhiteDiff` / `BookDepthBlackLimit` / `BookDepthWhiteLimit` を生やし、`BookMoves` 既定 200、
  `IgnoreBookPly` 既定 true にする)、`OptionsMapRef::count()` / `book_options_v2()` / `get_ref()`。
  `source/usi.cpp`: `USIEngine::set_engine()` で `add_options()` の前に profile を読む。
- 目的: ペタショック定跡 (peta15m、`.ybb` 頒布) を変換なしで使う。探索・評価には影響しない。
- `source/usi.cpp` (2026-09-12): `go ... searchmoves` の指し手を小文字化しない (upstream 同様の修正)。将棋 USI の打ち駒 `P*5e` は駒種が大文字なので、
  小文字化すると指し手が解釈できなかった。定跡候補の再評価ハーネス (tools/book_reeval.py) が searchmoves を使う。

## 定跡脱出 (二重逸脱) `ENABLE_BOOK_ESCAPE` (2026-09-12, task#81, report/57 §4-1)

- `source/engine/yaneuraou-engine/yaneuraou-search.cpp`: root で定跡にヒットしても、**相手の直前手が定跡 (優先定跡 + ペタショック) の
  最善手でない**ときは定跡手を指さず MultiPV (EscapeMultiPV) で探索し、最善から EscapeDelta [cp] 以内で「指した後の局面が定跡に無い」
  最上位の手を bestmove にする (相手を早く自力思考に入らせる。相手の定跡木は「相手側 = 最善のみ」で伸びているので 2 番手は薄い)。
  オプション: BookEscape / EscapeDelta / EscapeMaxPly / EscapeMultiPV / EscapeMaxCount (1 局の回数) / EscapeSide (white|black|both)。
  探索・評価は不変。相手の直前手の判定は StateInfo::lastMove (KEEP_LAST_MOVE) を使い、1 手戻して定跡を引く。
- `source/book/book.h` / `book.cpp`: `BookMoveSelector::has_position()` / `best_book_move16()` (副作用のない問い合わせ)。
- `-DENABLE_BOOK_ESCAPE` で有効 (既定は無効)。
- (2026-09-12 追記) `EscapeNoRejoinPly` (既定 6): 候補手の PV をその手数だけ辿り、途中で定跡に戻る (合流する) 候補は採らない。
  実測で単純な「子局面が定跡外」判定は 6 手以内にほぼ全件合流していたため。

## 千日手の価値を実局面への反復に限る `DrawValueHistoryOnly` (2026-09-14, task#80/#81, report/57 §7.1)

- `source/engine/yaneuraou-engine/yaneuraou-search.cpp` / `yaneuraou-search.h`: エンジンオプション `DrawValueHistoryOnly` (bool、既定 false)。
  true のとき、search / qsearch の千日手判定 (`is_repetition`) で返す値を、**同一局面の出現回数**で分ける:
  現局面が 2 回目 (木の中だけの反復、または実局面への最初の戻り) なら `DrawValueBlack/White` を乗せず `DrawValueTree` (int、既定 −2 =
  従来の既定 DrawValue と同じ、root 側 +/相手側 − の同じ規約) の値、3 回目以降 (`Position::repetition_count() >= 2`) なら従来どおり
  `draw_value()` (±DrawValue)。探索は 2 回目で打ち切るので、木の中で 3 回目に達するのは対局が既にその局面を 1 巡している場合に
  限られる = 「本当に千日手が目前」のときだけ価値が乗る。対局が 1 巡するまでは `DrawValueHistoryOnly=true` + 任意の DrawValue が
  既定と bestmove/score/PV まで完全一致することを固定ノード (1 スレッド) で確認 (2026-09-14)。
- 動機: 既定の実装は木の中の 2 回目の同一局面を ±DrawValue で評価するため、|DrawValue| が大きいと「相手はいつでも千日手にできる」線が
  全部 ±DrawValue になり、序中盤の棋風が壊れる (DrawValue ±100 で色別 ±250 Elo、千日手は増えない)。
  v1 (前回の出現が root 以前なら乗せる) は直前の実手を戻す線が全部 ±D になり root の手が動いたので、出現回数方式に変更 (2026-09-14)。
- `source/position.h` / `position.cpp`: `Position::repetition_count()` (遡り窓 `max_repetition_ply` 内の同一局面の回数。`ENABLE_QUICK_DRAW` 時は
  走査、無効時は `StateInfo::repetition_times`)。`is_repetition(ply)` の呼び出しは `is_repetition(ply, found_ply)` に置き換えたが、
  窓 (QUICK_DRAW は 16 手固定、それ以外は root まで + 4 回目) は同じ。
- 探索・評価は既定 (false) では不変 (固定ノードで bestmove/score 一致を確認)。

## threat 差分収集の exact 化 `THREAT_EXACT_DIFF` / 照合 `THREAT_DIFF_XCHECK` (2026-09-18, task#59, report/52 §20)

- `eval/nnue/features/threat.cpp`: 差分収集器を 2 つに分離。`collect_piece_diff_set` (従来: 影響 attacker の利き先を prev/now 両占有で
  全列挙して対称差分) と `collect_piece_diff_exact` (新: 1 手で変わる対 = A 動いた駒が攻撃側 / B 動いた駒が被弾側 / C 取られた駒が攻撃側 /
  D 取られた駒が被弾側 / E from・to を通る長い利きの駒の被弾集合のビットボード差分、の 5 群を直接出す。ソート・マージ・不変対の列挙なし)。
- 既定は従来の set 版 (既存ビルドはビット同一)。`-DTHREAT_EXACT_DIFF` で exact 版。`-DTHREAT_DIFF_XCHECK` は両方を走らせて
  removed/added を集合比較し、不一致なら局面と手を出して即死 (研究ビルド)。`-DTHREAT_DIFF_STATS` に両版の rdtsc サイクル集計を追加。
- 照合: xcheck ビルドで 20 局面 × 200k ノード (王者 tsfnn-526-q128 @13) = 3,547,135 回の収集で不一致 0。
  サイクル (L77 対局と同居、Core Ultra 7 265K): set 945 / exact 382 cycles/収集。
- idle NPS (2026-09-18、nps_bench ABBA 40 局面 × 400k、2 回): 王者 exact/現行 = ×1.0795 / ×1.0769、classic lite = ×1.1002 / ×1.0986 → 採用
  (新規ビルドの既定 define に `-DTHREAT_EXACT_DIFF` を合成)。

## narrow threat slot `NNUE_THREAT_NARROW_COLS=N` + FT 計測の追加 define (2026-09-18, task#59 ⑤/⑥, report/52 §21.2〜21.3)

- `eval/nnue/nnue_feature_transformer.h`:
  - `NNUE_THREAT_NARROW_COLS=N`: 列マスクで訓練した SFNN halfka2t ネット (threat 行の重みが列 [0,N)∪[512,512+N) 以外で 0) 専用の推論経路。
    threat (kRefreshTriggers[0] = kNone) の行を 2N 幅で格納 (`row_off()`、`kWeightsCount`)、accumulator の narrow slot は先頭 2N 要素だけ
    使い (`slot_width()`)、バイアスは full slot (`kBiasSlot`) へ、Transform は各半分の先頭 N 列にだけ narrow slot を足す。nn.bin は
    既存の全幅 LEB128 のままロード時にストリーム復号して圧縮格納し (`read_leb_128_sink`)、スライス外に非零があれば FileReadError。
    通常ビルドでは kBiasSlot = 0 / slot_width = kHalf で従来と同一 (探索一致 20/20)。
  - `FT_STAT_NO_ROW_TIMING` (行ごと rdtsc を外す) / `FT_STAT_ALIAS_THREAT_ROWS` (threat 行 index を 4096 で畳む、計測専用・評価は変わる)。
- 検証 (09-18): exact2 (リファクタ後) vs hist2 = 探索一致 20/20 (王者ネット)、narrow128 vs exact2 = 探索一致 20/20 (slice128 sb46 ネット、
  q128 export)、narrow128 に王者ネットを読ませると 134,395,171 個の非零を検出して読込失敗 (負のテスト OK)。

## narrow threat slot のブロック疎化 `NNUE_THREAT_NARROW_BLOCKS=B` (2026-09-18, task#59 ⑦, report/52 §21.5)

- `eval/nnue/nnue_feature_transformer.h`: threat 行 t (Head 側の index − kTailRows) はブロック b = t % B を使い、列 [b·N, b·N+N) ∪
  [512 + b·N, …) にだけ重みを持つ (訓練側 BulletOu `--threat-slice-blocks B` の列マスクと同じ規則)。格納は B=1 と同じ 2N 幅
  (`row_width()`)、ロード時のスライス外検査も行ごとのブロックで行う。B>1 では accumulator の narrow slot を full 幅で持ち
  (`slot_width()` = kHalf)、行の加減算は `apply_narrow_row<kAdd>()` が各半分のブロック位置へ直接足す (Transform 側の narrow 加算は不要)。
  B=1 (従来 narrow) と非 narrow ビルドは変更なし。
- 検証 (09-18 12:00): narrow128 (B=1、リファクタ後) vs exact2 = 探索一致 20/20 (slice128 sb46 ネット)、narrow128-b4 (B=4) vs exact2 =
  探索一致 **20/20** (slice128 sb46 の threat 行を B=4 規則でマスクした合成ネット `synth-b4-q128`、tmp/synth_block_state.py)。
