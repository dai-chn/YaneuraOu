"""512ch NNUE 配布パッケージ生成。--eval 差替えで再生成可能。

パッケージツリー:
  <name>-<version>/
    <name>.exe            (dispatcher)
    engine/YaneuraOu-<ISA>.exe    (各 SIMD)
    engine/eval/nn.bin
    README.txt / MODEL.txt / SOURCE.txt / LICENSE-GPLv3.txt / LICENSE-bullet-MIT.txt

静的リンク (BUILD_NOTES.md) なので DLL 同梱は不要 (dlls=[])。
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import click

from build_engine import build_engine

TARGETS_DEFAULT = ["AVX512", "AVX2", "SSE42"]
TEMPLATES = Path(__file__).parent / "templates"


def assemble_tree(staging: Path, *, exes: dict[str, Path], dispatcher: Path,
                  eval_bin: Path, dlls: list[Path], name: str) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "engine" / "eval").mkdir(parents=True)
    shutil.copy2(dispatcher, staging / f"{name}.exe")
    for isa, exe in exes.items():
        shutil.copy2(exe, staging / "engine" / f"YaneuraOu-{isa}.exe")
    for dll in dlls:
        shutil.copy2(dll, staging / "engine" / dll.name)
    shutil.copy2(eval_bin, staging / "engine" / "eval" / "nn.bin")


def assemble_single(staging: Path, *, exe: Path, eval_bin: Path, name: str) -> None:
    """単一 ISA・ディスパッチャ無しのツリー。exe を root 直下、eval を root/eval に置く。"""
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "eval").mkdir(parents=True)
    shutil.copy2(exe, staging / f"{name}.exe")
    shutil.copy2(eval_bin, staging / "eval" / "nn.bin")


def write_engine_name(staging: Path, engine_name: str, author: str) -> None:
    """engine_name.txt を書く (1行目=id name, 2行目=著者)。YaneuraOu が cwd 相対で読む。"""
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "engine_name.txt").write_text(f"{engine_name}\n{author}\n", encoding="utf-8")


def write_docs(staging: Path, meta: dict) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    # engine_name があれば単一exe用 README、無ければディスパッチャ用 README。
    readme_tmpl = "README-single.txt.tmpl" if meta.get("engine_name") else "README.txt.tmpl"
    for tmpl, out in [(readme_tmpl, "README.txt"),
                      ("MODEL.txt.tmpl", "MODEL.txt"),
                      ("SOURCE.txt.tmpl", "SOURCE.txt")]:
        text = (TEMPLATES / tmpl).read_text(encoding="utf-8")
        for k, v in meta.items():
            text = text.replace("{{" + k + "}}", str(v))
        (staging / out).write_text(text, encoding="utf-8")
    # ライセンスを丸ごとコピー (存在すれば)
    fork = Path(__file__).resolve().parents[1]  # YaneuraOu/
    gpl = fork / "Copying.txt"
    if gpl.exists():
        shutil.copy2(gpl, staging / "LICENSE-GPLv3.txt")
    bullet_mit = fork.parent / "bullet-shogi" / "LICENSE"
    if bullet_mit.exists():
        shutil.copy2(bullet_mit, staging / "LICENSE-bullet-MIT.txt")


def _git_commit(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip()[:12] if r.returncode == 0 else "UNKNOWN"


@click.command()
@click.option("--eval", "eval_bin", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--arch", default="halfkp-256x2-32-32", show_default=True)
@click.option("--name", default="ShogiNNUE-512ch", show_default=True, help="パッケージ/exe のベース名 (空白なし推奨)")
@click.option("--version", default="v1", show_default=True)
@click.option("--targets", default=",".join(TARGETS_DEFAULT), show_default=True)
@click.option("--repo-url", default="https://github.com/<USER>/<FORK>", show_default=True)
@click.option("--commit", "commit_override", default=None, help="SOURCE.txt に書く対応コミット (既定: フォーク HEAD から取得)")
@click.option("--out", default="dist", type=click.Path(path_type=Path), show_default=True)
@click.option("--skip-build", is_flag=True, help="既存 builds/ を使う (再ビルドしない)")
@click.option("--engine-name", default=None, help="USI id name (engine_name.txt に書く)")
@click.option("--author", default="tarakojo", show_default=True)
@click.option("--jp-name", default="", help="日本語エンジン名 (README 表示用)")
@click.option("--dispatcher/--no-dispatcher", "use_dispatcher", default=None,
              help="ディスパッチャ同梱の有無 (既定: target が1個なら no-dispatcher)")
@click.option("--model-note", default="", help="MODEL.txt に書く強さ等の説明")
@click.option("--arch-desc", default=None, help="MODEL.txt のアーキ表記 (既定: --arch から導出)")
def main(eval_bin, arch, name, version, targets, repo_url, commit_override, out, skip_build,
         engine_name, author, jp_name, use_dispatcher, model_note, arch_desc):
    fork = Path(__file__).resolve().parents[1]            # YaneuraOu/
    repo_root = fork.parent
    source_dir = fork / "source"
    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    if use_dispatcher is None:
        use_dispatcher = len(target_list) > 1
    if arch_desc is None:
        arch_desc = "HalfKP_" + arch.removeprefix("halfkp-")

    # 各 ISA をビルド (or 既存を使う)
    exes: dict[str, Path] = {}
    for isa in target_list:
        dest = repo_root / "builds" / "yaneuraou" / f"{arch}-{isa}" / "YaneuraOu.exe"
        if not skip_build:
            built = build_engine(arch, isa, source_dir=source_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built, dest)
        if not dest.exists():
            raise click.ClickException(f"engine binary not found: {dest} (--skip-build 指定時は先にビルドが要る)")
        exes[isa] = dest

    staging = Path(out) / f"{name}-{version}"
    if use_dispatcher:
        disp = repo_root / "builds" / "dist" / f"{name}.exe"
        disp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["bash", str(fork / "dispatcher" / "build.sh"), str(disp)], check=True)
        assemble_tree(staging, exes=exes, dispatcher=disp, eval_bin=Path(eval_bin), dlls=[], name=name)
    else:
        if len(target_list) != 1:
            raise click.ClickException("--no-dispatcher は target が1個のときのみ有効")
        assemble_single(staging, exe=exes[target_list[0]], eval_bin=Path(eval_bin), name=name)

    if engine_name:
        write_engine_name(staging, engine_name, author)

    meta = {
        "name": name, "version": version,
        "model": Path(eval_bin).parent.name,
        "engine_name": engine_name or "", "jp_name": jp_name,
        "model_note": model_note, "arch_desc": arch_desc,
        "repo_url": repo_url, "commit": commit_override or _git_commit(fork), "arch": arch,
    }
    write_docs(staging, meta)

    zpath = Path(out) / f"{name}-{version}-win.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(staging.parent))
    click.echo(f"packaged -> {zpath}")


if __name__ == "__main__":
    main()
