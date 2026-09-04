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

### 2. NNUE アーキテクチャ `halfkp_512x2-16-32` の追加

配布版の評価関数 (`eval/nn.bin`) が使うアーキテクチャ。
`source/eval/nnue/architectures/halfkp_512x2-16-32.h` (新規) と
`source/eval/nnue/nnue_architecture.h` / `source/Makefile` の分岐。

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
| `halfkp_768x2-8-32.h` / `halfkp_768x2-16-32.h` | FT 幅 768 の検証 |
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
