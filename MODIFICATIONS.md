# Modifications from upstream YaneuraOu

Upstream: https://github.com/yaneurao/YaneuraOu
Base commit: `cc73ac44ad1433463df73fc6c97c06fd27c5d266` (2026-05-09、`UPSTREAM.md` 参照)
License: GPLv3 (`Copying.txt`)

本フォークは上流 YaneuraOu を基点に、将棋NNUE研究プロジェクトの配布・実験用に以下を改変・追加した
(GPLv3 §5 の改変告知)。**既定の NNUE ビルド (`YANEURAOU_ENGINE_NNUE`, HalfKP_256x2-32-32, CReLU) の
探索・評価の挙動を変える変更は wide-path 修正を除いて無い** — 実験系はすべてコンパイルフラグの後ろ、
または対局中に呼ばれない追加サブコマンド。

## 配布・移植性

- `source/eval/nnue/evaluate_nnue.cpp` / `source/misc.cpp` / `source/misc.h`:
  評価ファイル (nn.bin) のロードを **wide(UTF-16) Windows API 化** (`Directory::ReadBinaryFolderRelativeFileW`
  = `GetModuleFileNameW` + `CreateFileW`)。ANSI コードページ (例 CP932) で表現できない文字を含むパス
  (韓国語/絵文字/混在スクリプト) でも評価関数を読めるようにする。**探索・評価の数値は不変** (ロード経路のみ変更)。
- `dispatcher/`: CPU の SIMD を判定し `engine\YaneuraOu-<ISA>.exe` を透過起動するランチャ (YaneuraOu 本体とは
  独立したプログラム。CreateProcessW で子を起こし stdio を継承 = USI プロトコルに介入しない)。
- `packaging/`: 配布パッケージ生成スクリプト (改変 YaneuraOu を 3 SIMD でビルドし、ディスパッチャ・評価関数・
  ライセンスと共に zip 化する)。GPLv3 の「ビルド/インストールを制御するスクリプト」に相当。

## 実験用アーキ variant (コンパイルフラグ後ろ、既定ビルド非影響)

- `source/eval/nnue/architectures/halfkp_768x2-8-32.h` (新規)、`halfkp_256x2-32-32-screlu.h` (新規)、
  `halfkp_256x2-32-32-pairwise.h` (新規)。
- `source/eval/nnue/nnue_architecture.h`: 上記 arch の分岐 (`EVAL_NNUE_HALFKP256_SCRELU` / `_PAIRWISE`)。
- `source/eval/nnue/nnue_feature_transformer.h`: SCReLU (`NNUE_FT_SCRELU`) / pairwise (`NNUE_FT_PAIRWISE`)
  の FT 変換と hash マーカー (AVX2/scalar のみ)。
- `source/eval/nnue/evaluate_nnue.cpp`: SCReLU/pairwise ビルドで nn.bin の hash 不一致を hard error にする
  (`#if defined(NNUE_FT_SCRELU) || defined(NNUE_FT_PAIRWISE)` の後ろ、異種ネット混載の拒否)。
- `source/misc.cpp`: `config_info()` に 768x2 の arch 名を追加。
- `source/Makefile`: 上記 arch 用の preset。

## データ生成・検証ツール

- `source/learn/filter_quiet.cpp` (新規) + `source/usi.cpp`: `filter_quiet` USI サブコマンド
  (静止局面フィルタ、データセット生成用。`USE_SFEN_PACKER` ガード、対局中は呼ばれない)。
- `source/engine/yaneuraou-engine/yaneuraou-search.cpp`: `trace_eval()` が静的評価値を `eval = <値>` 形式で
  出力 (元は空スタブ)。`eval` USI コマンド実行時のみ動作、探索挙動は不変。export 検証 (`tools/nnue_eval.py`) 用。
