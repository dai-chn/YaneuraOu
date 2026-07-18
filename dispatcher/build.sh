#!/usr/bin/env bash
# ディスパッチャをビルドする。
# usage: build.sh <out_exe_path>
# MSYS2 clang64 の clang++ が要る (PATH に /c/msys64/clang64/bin)。
set -euo pipefail
export PATH="/c/msys64/clang64/bin:$PATH"
OUT="${1:-ShogiNNUE.exe}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/dispatcher.cpp"

# 静的リンク優先 (エンジン本体と同様 UCRT 以外の DLL 依存を無くす)。失敗時は動的にフォールバック。
if clang++ -O2 -std=c++17 -o "$OUT" "$SRC" -static -lkernel32 2>/dev/null; then
  :
else
  clang++ -O2 -std=c++17 -o "$OUT" "$SRC" -lkernel32
fi
echo "built dispatcher -> $OUT"
