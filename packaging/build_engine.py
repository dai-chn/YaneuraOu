"""単一 (arch, target_cpu) の YaneuraOu ビルドヘルパ。フォーク自己完結。"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

EVAL_TYPE_MAP = {
    "halfkp-256x2-32-32": "YANEURAOU_ENGINE_NNUE",                     # 64MB CReLU 生徒
    "halfkp-512x2-16-32": "YANEURAOU_ENGINE_NNUE_HALFKP_512X2_16_32",  # 128MB 幅512 生徒 (hybrid-fuka-512x2)
}
EXTRA_CPPFLAGS_MAP: dict[str, str] = {}  # screlu/pairwise は非対象 (CReLU のみ)

# Windows の subprocess は実行ファイル解決に env['PATH'] でなく親プロセスの PATH を使うため、
# mingw32-make.exe は絶対パスで指定する (Makefile が呼ぶ clang++ は make の env で解決される)。
CLANG64_BIN = r"C:\msys64\clang64\bin"
MAKE_EXE = CLANG64_BIN + r"\mingw32-make.exe"


def _make_args(arch: str, target_cpu: str, jobs: int) -> list[str]:
    edition = EVAL_TYPE_MAP[arch]
    args = [
        MAKE_EXE,
        f"-j{jobs}",
        "tournament",
        "COMPILER=clang++",
        f"YANEURAOU_EDITION={edition}",
        f"TARGET_CPU={target_cpu}",
    ]
    extra = EXTRA_CPPFLAGS_MAP.get(arch)
    if extra:
        args.append(f"EXTRA_CPPFLAGS={extra}")
    return args


def build_engine(
    arch: str,
    target_cpu: str,
    *,
    source_dir: Path,
    jobs: int = 8,
    clean: bool = True,
) -> Path:
    """arch/target_cpu をビルドし、生成された exe の Path を返す。失敗時 RuntimeError。"""
    source_dir = Path(source_dir)
    env = dict(os.environ)
    env["PATH"] = CLANG64_BIN + os.pathsep + env.get("PATH", "")
    if clean:
        subprocess.run([MAKE_EXE, "clean"], cwd=str(source_dir), env=env, check=False)
    proc = subprocess.run(_make_args(arch, target_cpu, jobs), cwd=str(source_dir), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"make failed ({proc.returncode}) for {arch}/{target_cpu}")
    cands = list(source_dir.glob("YaneuraOu-*")) + list(source_dir.glob("YaneuraOu*.exe"))
    cands = [c for c in cands if c.suffix in (".exe", "") and c.is_file()]
    if not cands:
        raise RuntimeError("built exe not found under source_dir")
    return max(cands, key=lambda p: p.stat().st_mtime)
