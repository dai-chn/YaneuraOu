# ビルド・パッケージング ノート

## 静的リンク可否 (Task 0.2, 2026-07-05)

**結論: LINK_MODE = static。ランタイム DLL の同梱は不要。**

YaneuraOu の Makefile (clang64/`tournament`) は既定でリンク段に `-static` を含み、
`libc++` / `libunwind.a` / `libclang_rt.builtins` / `libmingwex` 等を静的リンクする。

AVX2 ビルド (`builds/yaneuraou/halfkp-256x2-32-32/YaneuraOu.exe`) の PE インポート DLL
(`objdump -p`) は以下のみ:

- `api-ms-win-crt-*.dll` (Universal CRT フォワーダ。**Windows 10 以降に標準搭載**)
- `KERNEL32.dll` (常時存在)

→ **msys2/clang64 のランタイム DLL (`libc++.dll` / `libunwind.dll` / `libwinpthread-1.dll` 等)
への依存はゼロ**。`ldd` でも msys/clang64 DLL 依存なしを確認。

### パッケージングへの反映
- `package_engine.py` の `dlls` は空リストのまま (DLL を `engine/` に同梱しない)。
- ディスパッチャ (`dispatcher/build.sh`) も同様に `-static` を試み、失敗時のみ動的にフォールバック。

### 留意
- UCRT (`api-ms-win-crt-*`) は Windows 10/11 に標準搭載。配布対象は Windows 10+ とする
  (Windows 7/8 で UCRT 未導入の場合のみ Microsoft の UCRT 更新が別途必要だが、対象外)。
