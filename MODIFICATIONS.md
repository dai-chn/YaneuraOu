# Modifications from upstream YaneuraOu

Upstream: https://github.com/yaneurao/YaneuraOu
Base commit: `cc73ac44ad1433463df73fc6c97c06fd27c5d266` (2026-05-09、`UPSTREAM.md` 参照)
License: GPLv3 (`Copying.txt`)

本ツリーは上流 YaneuraOu を基点に、将棋エンジン **Kisou Engine (棋想)** の配布バイナリを
ビルドするために改変・追加したもので、配布バイナリの対応ソース (GPLv3 §6) である。
以下は改変の告知 (GPLv3 §5a)。日付は各改変を行った日。

## 配布版の挙動に直接効く改変

### 1. 探索パラメータの再調整 (SPSA)

`source/engine/yaneuraou-engine/yaneuraou-search.cpp`:
razoring / futility / null move / LMR reduction / aspiration window / ProbCut /
SEE margin / singular extension / IIR の各定数 **32 個**を `TUNABLE_PARAM` マクロで定義し、
SPSA (Fishtest 方式) による最適化結果を既定値として反映した (2026-08、2026-11-05 更新)。

```
#if defined(ENABLE_SEARCH_TUNE)
    #define TUNABLE_PARAM(name, val) int name = val;      // USI オプションとして可変
#else
    #define TUNABLE_PARAM(name, val) constexpr int name = val;  // 焼き込み (配布版)
#endif
```

- `ENABLE_SEARCH_TUNE` 未定義 (= 配布ビルド) では `constexpr` のままなので、
  上流と同じくコンパイル時定数として畳み込まれる。
- `ENABLE_SEARCH_TUNE` 定義時のみ USI オプションとして露出し、Stockfish の `TUNE()` 機構で
  SPSA から駆動できる。`source/tune.cpp` は Fishtest 用パラメータ CSV の出力先を
  `std::cout` → `std::cerr` に変更 (stdout は USI プロトコルのストリームなので非 USI 行を混ぜない)。

### 2. NNUE 評価関数のアーキテクチャ

配布版 v2 の評価関数 (`eval/nn.bin`) は SFNN 型 **HalfKA2 + Threat** (利き当たり特徴)、
隠れ層 1024-7-64、レイヤースタック 9 (双方の玉の段 3×3) で、以下を追加した。

- `source/eval/nnue/features/threat.h/.cpp` (新規、2026-08-21〜): **Threat 特徴**。
  駒の実利きが駒に当たっている関係 (玉除外、9 クラス × 敵味方、空盤幾何で圧縮、216,720 次元)。
  テーブルの FNV-1a チェックサムを焼き込み、不一致なら起動時に停止する。
  `source/eval/nnue/features/threat_ka2.h` (新規): HalfKA2 と組にするためのバリアント (ハッシュ値のみ変更)。
- `source/eval/nnue/features/threat_diff.h` (新規、2026-09-05): 駒レベルの threat 差分 (対の増減) の共有型。
- **差分更新** (2026-08-22〜09-19): `kRefreshTrigger = kNone` で毎手 `AppendChangedIndices` により
  変化した対だけを更新する。影響する対の収集器は 2 つ (`collect_piece_diff_set` = 影響 attacker の
  利き先を前後の占有で全列挙して対称差分、`collect_piece_diff_exact` = 1 手で変わる対を手の幾何から直接出す。
  `-DTHREAT_EXACT_DIFF` で後者、配布ビルドは後者)。両視点の index は BLACK 視点の呼び出しで 1 パスで作り
  WHITE 視点はキャッシュから写す。`-DTHREAT_NAIVE_REBUILD` で毎手全再構築に戻せる。
- `source/config.h`: threat 系 edition で `KEEP_LAST_MOVE` を定義 (差分更新が直前の指し手を要する)。
  `source/Makefile`: edition 名に `_halfka2t_` を含むとき `-DNNUE_SFNN_HALFKA2T` を定義する。
- `source/eval/nnue/features/feature_set.h` (2026-09-19): 特徴クラスが差分更新の前提 define を欠いて
  黙って全再構築に落ちているとき (`kNaiveFallback`) は、その特徴を使う edition で
  `nnue_feature_transformer.h` の `static_assert` によりコンパイルエラーにする。
- `source/eval/nnue/architectures/nnue_arch_gen.py` / `SFNNwoPSQT_halfka2t_1024-7-64-ls9.h`:
  入力特徴 `halfka2t` (`FeatureSet<ThreatKa2, HalfKA2<kFriend>>`) を生成器に追加し、生成したヘッダを同梱。
  レイヤースタック接尾辞 `ls<K>[p<P>]` (K = 玉位置 bucket 数、P = 進行度 bucket 数) を追加。
- 旧配布版 (v1、2026-08〜) の HalfKP 型アーキテクチャ `halfkp_768x2-16-32.h` 等もそのまま残している
  (「実験用アーキ variant」の節)。

### 3. NNUE 特徴変換器 (FT) の高速化

`source/eval/nnue/nnue_feature_transformer.h` (+ `evaluate_nnue.cpp`):

- **int8 行** `NNUE_FT_INT8_ROWS` (2026-09-18〜19): FT 重みを nn.bin の値のまま int8 で格納し、
  ロード時の ×2 を掛けない (accumulator は 1/2 スケール、Transform で補正)。|v| > 127 の重みは clip し、
  差分を行ごとの残差表 (行の先頭バイト −128 を番兵) で足し戻すので int16 経路とビット一致する。
  `WriteParameters` は未対応。
- **差分更新のタイル化** `NNUE_FT_TILED` (2026-09-18、VECTOR ビルドの既定): accumulator をレジスタ幅の
  タイルに分け、全行の加減算をタイル単位で行ってから 1 回だけ書き戻す。`prev_accumulator` の値コピーを
  参照に変更。整数加算の順序非依存性により結果はビット一致。
- **accumulator キャッシュ** `ENABLE_ACC_CACHE` (2026-09-08): 玉移動トリガの accumulator を
  視点 × 自玉升ごとの thread_local キャッシュとの集合差分で作る (Finny table)。ネット再読込で無効化。
- `source/eval/nnue/evaluate_nnue.cpp` / `yaneuraou-search.cpp`: **差分連鎖の維持** `ENABLE_ACC_CHAIN_FIX`
  (2026-08-19): `search()` / `qsearch()` の王手局面で `evaluate_with_no_return()` を呼び、
  accumulator の差分計算を途切れさせない。評価値を返さないので探索結果は変わらない。
- `ENABLE_FT_TRAFFIC_STAT` (計測用、配布ビルドには含まれない): FT 行の読み書きの内訳カウンタ。

### 4. 層別の重みスケールと量子化タグ (2026-08-27〜28)

- `source/eval/nnue/layers/affine_transform.h` / `affine_transform_sparse_input.h` / `clipped_relu.h` /
  `clipped_relu_explicit.h` / `sqr_clipped_relu.h`: テンプレート引数 `WeightScaleBits` で層ごとに
  重みスケールを切替可能に。`-DNNUE_ROUND_SHIFT` で床シフトを四捨五入に (検証用、既定 off)。
- classic 型ヘッダは `NNUE_L1_SCALE_BITS`、SFNN 型は `NNUE_SFNN_L1_SCALE_BITS` (既定 6) で L1 の
  スケールを切替 (7 = QB 128。配布ビルドは 7)。`layers/affine_transform_sparse_input_i16.h` (新規):
  L1 の int16 重み版 (`-DNNUE_L1_INT16`)。
- `source/eval/nnue/evaluate_nnue.cpp` `QuantTag()`: nn.bin の description 末尾の `;L1QB=<qb>/<i8|i16>` を
  ビルド設定と照合し、不一致なら読み込みを拒否する (ネットワークハッシュはスケールビットに依存しないため)。

### 5. 進行度 bucket の取り込み (2026-10-08)

`source/eval/nnue/evaluate_nnue.{h,cpp}` / `source/position.h`: 上流 9.80git の進行度 (progress) bucket
(commit 9a0df6a5 / e47d460f / c1b80eaa) を cherry-pick。SFNN のレイヤースタックを
「玉位置 bucket × 進行度 bucket」で選ぶ (手駒 bucket `NNUE_SFNN_HAND_BUCKETS` 64/256/1024 と玉位置 bucket 81/441/841 の変種も同時に取り込んだ)。
配布版の edition (`ls9` = 進行度 bucket 1) では無効で、
評価値・探索は従来と同一。

### 6. 定跡

- `source/book/book.h` / `book.cpp` / `makebook.cpp` / `makebook2025.cpp`
  (2026-09-10): 上流 master の定跡サブシステムに置換。やねうら王バイナリ定跡 DB (`.ybb`) の読み込み、
  `.db` が無ければ `.ybb` への fallback、優先定跡 (`user_book1-000.db` 等) の複数保持。
  `makebook2015.cpp` (旧 makebook コマンド群) は上流同様に削除。
  `source/extra/sfen_packer.cpp` / `position.h`: `PackedSfen::flip()` / `flipped()` (FlippedBook 用)。
  `source/usioption.{h,cpp}` / `usi.cpp`: `engine_option_profile.txt` の `BOOK_OPTIONS=V2` で
  上流 V2 の定跡オプション群を生やす `read_engine_option_profile()`。
- `source/book/book.cpp` `BookMoveSelector::probe_impl` (2026-11-11〜14): **同一局面の繰り返しに対する定跡の抑止**。
  (a) 定跡の候補手のうち、指した後の局面がこの対局で既に出現しているもの (盤面と手駒が同一) を候補から外し、
  残らなければ定跡を使わず探索に任せる。(b) root 局面そのものがこの対局で既に出現しているなら定跡を使わず
  探索に任せる (相手が可逆手で同じ局面を作り直している = 千日手狙い)。ただし対局開始直後の往復手で初期局面に戻った場合 (初期局面との 4 手以内の一致) は対象外。局面キーで引く定跡は千日手を見ないため。
  判定は StateInfo の履歴を対局開始まで 2 手ずつ遡って盤面と手駒を直接比べる。
- `source/usi.cpp` (2026-09-12): `go ... searchmoves` の指し手を小文字化しない (将棋 USI の打ち駒 `P*5e` は
  駒種が大文字。上流と同様の修正)。

### 7. 時間管理

`source/engine/yaneuraou-engine/yaneuraou-search.cpp` `start_searching()` の末尾 (2026-11-13):
定跡・宣言勝ち・合法手なしで探索を skip したとき、`bestPreviousScore` / `bestPreviousAverageScore` を
rootMoves[0] (既定値 −∞) から保存せず「前回評価なし」(VALUE_INFINITE) にする。従来は定跡を抜けた
最初の 1 手で fallingEval が「−∞ からの急上昇」と見て時間係数を下限に張り付けていた。
探索・評価は不変で、変わるのは定跡直後の 1 手の時間配分だけ。

## 配布・移植性

- `source/eval/nnue/evaluate_nnue.cpp` / `source/misc.cpp` / `source/misc.h`:
  評価ファイル (nn.bin) のロードを wide (UTF-16) Windows API 化
  (`Directory::ReadBinaryFolderRelativeFileW` = `GetModuleFileNameW` + `CreateFileW`)。
  ANSI コードページで表現できない文字を含むパスでも評価関数を読める。探索・評価の数値は不変。
- `dispatcher/`: CPU の SIMD を判定し `engine\YaneuraOu-<ISA>.exe` を透過起動するランチャ
  (本体とは独立したプログラム。CreateProcessW で子を起こし stdio を継承)。
- `packaging/`: 配布パッケージ生成スクリプト (GPLv3 の「ビルド/インストールを制御するスクリプト」に相当)。
- `jni/Android.mk`: `filter_quiet.cpp` / `policybook.cpp` の追加と `makebook2015.cpp` の削除 (2026-09-19)。`source/YaneuraOu.vcxproj(.filters)` からも `makebook2015.cpp` を外した。
- `source/engine/yaneuraou-engine/yaneuraou-search.cpp`: `EvLog::ctx` を非 NNUE エディションでも定義
  (MATERIAL 等でのリンクエラー修正、2026-09-24)。

## 実験用アーキ variant (コンパイルフラグ後ろ、配布ビルド非影響)

`source/eval/nnue/architectures/` に以下を追加し、`nnue_architecture.h` と `Makefile` に分岐を追加:

| ヘッダ | 用途 |
|---|---|
| `halfkp_512x2-8-64.h` / `halfkp_512x2-32-32.h` / `halfkp_512x2-16-32.h` / `halfkp_512x2-16-32-screlu.h` | 512 系の変種 (旧配布アーキ 2026-08-15〜08-30 を含む) |
| `halfkp_768x2-8-32.h` / `halfkp_768x2-16-32.h` / `halfkp_768x2-16-64.h` | FT 幅 768 (768x2-16-32 = 配布版 v1) |
| `halfkp_1024x2-8-32.h` / `halfkp_1024x2-8-64.h` | FT 幅 1024 |
| `halfka_512x2-16-32.h` | HalfKA 入力 |
| `halfkp_256x2-32-32-screlu.h` / `-pairwise.h` | 256 系の活性化 variant |
| `halfkp_threat_512x2-16-32.h` / `halfkp_threatlite_512x2-16-32.h` | HalfKP + Threat / ThreatLite (`features/threat_lite.*`: Threat の縮約版 26,244 次元) |
| `sfnnwop-1536.h` | SFNN 型 (upstream 付属) |
| `SFNNwoPSQT_halfka2_1024-7-64-ls9.h` | SFNN HalfKA2 1024-7-64 (threat なし) |

- `source/eval/nnue/nnue_feature_transformer.h`: SCReLU (`NNUE_FT_SCRELU`) / pairwise (`NNUE_FT_PAIRWISE`) の
  FT 変換と hash マーカー (AVX2/scalar のみ)。
- `source/eval/nnue/evaluate_nnue.cpp`: SCReLU/pairwise ビルドで nn.bin の hash 不一致を hard error にする。
- `source/misc.cpp`: `config_info()` に追加 arch 名。

## データ生成・検証・計測ツール (対局中は呼ばれない / 配布ビルドには含まれない)

- `source/learn/filter_quiet.cpp` (新規) + `source/usi.cpp`: `filter_quiet` USI サブコマンド
  (静止局面フィルタ、データセット生成用。`USE_SFEN_PACKER` ガード)。
- `source/engine/yaneuraou-engine/yaneuraou-search.cpp`: `trace_eval()` が静的評価値を `eval = <値>` 形式で出力
  (`eval` USI コマンド実行時のみ)。
- eval 呼び出し統計の計測パッチ (`EVAL_LOG_PATH` 未設定なら完全無効)。
- `source/eval/nnue/evaluate_nnue.cpp`: 合成 small eval ゲートのシミュレータ `EvalGateSim`
  (`ENABLE_EVAL_GATE_SIM` を定義したビルドにのみ存在)。
- `source/history.h` + `yaneuraou-search.cpp`: material correction history (`ENABLE_MATERIAL_CORRHIST`、定義時のみ)。
- `ENABLE_PROBE_EVAL_TWICE` / `ENABLE_FT_TRAFFIC_STAT`: 計測用 (定義時のみ)。

## 配布バイナリのビルド

```
make tournament OS=Windows_NT COMPILER=clang++ TARGET_CPU=AVX2 \
     YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_SFNNwoPSQT_halfka2t_1024-7-64-ls9 \
     EXTRA_CPPFLAGS="-DNNUE_SFNN_L1_SCALE_BITS=7 -DENABLE_ACC_CHAIN_FIX -DENABLE_ACC_CACHE -DTHREAT_EXACT_DIFF -DNNUE_FT_INT8_ROWS"
```

(MSYS2 clang64、静的リンク。`OS=Windows_NT` を明示しないと DLL 依存の exe になる。)
配布版 v1 (HalfKP 768x2-16-32) は `YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_HALFKP_768X2_16_32`、
`EXTRA_CPPFLAGS=-DNNUE_L1_SCALE_BITS=7`。
