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


def _render(tmpl_name: str, meta: dict) -> str:
    text = (TEMPLATES / tmpl_name).read_text(encoding="utf-8")
    for k, v in meta.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def _copy_gpl(staging: Path) -> None:
    gpl = Path(__file__).resolve().parents[1] / "Copying.txt"  # YaneuraOu/Copying.txt
    if gpl.exists():
        shutil.copy2(gpl, staging / "LICENSE-GPLv3.txt")


def write_docs(staging: Path, meta: dict) -> None:
    """standalone 配布用の docs 一式 (README/MODEL/SOURCE + ライセンス)。"""
    staging.mkdir(parents=True, exist_ok=True)
    # engine_name があれば単一exe用 README、無ければディスパッチャ用 README。
    readme_tmpl = "README-single.txt.tmpl" if meta.get("engine_name") else "README.txt.tmpl"
    for tmpl, out in [(readme_tmpl, "README.txt"),
                      ("MODEL.txt.tmpl", "MODEL.txt"),
                      ("SOURCE.txt.tmpl", "SOURCE.txt")]:
        (staging / out).write_text(_render(tmpl, meta), encoding="utf-8")
    _copy_gpl(staging)
    bullet_mit = Path(__file__).resolve().parents[2] / "bullet-shogi" / "LICENSE"
    if bullet_mit.exists():
        shutil.copy2(bullet_mit, staging / "LICENSE-bullet-MIT.txt")


def write_bundle_docs(staging: Path, meta: dict) -> None:
    """GUI バンドル用の最小 docs: GPLv3 ライセンス + SOURCE + nn.bin NOTICE のみ。

    README/MODEL/bullet-MIT は出さない (使い方は GUI が提供、研究情報は非公開、bullet は
    配布バイナリに含まれない)。GPLv3 の義務 (ライセンス文 + 対応ソース入手法) だけ残す。
    """
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "SOURCE.txt").write_text(_render("SOURCE.txt.tmpl", meta), encoding="utf-8")
    _copy_gpl(staging)
    if meta.get("nn_notice"):
        (staging / "NOTICE.txt").write_text(str(meta["nn_notice"]) + "\n", encoding="utf-8")


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
@click.option("--bundle", is_flag=True, help="GUI バンドル用の最小構成 (README/MODEL/bullet-MIT を出さず GPLv3+SOURCE+NOTICE のみ)")
@click.option("--nn-notice", default=None, help="nn.bin の権利表記1行 (--bundle 時 NOTICE.txt に。既定は author から生成)")
def main(eval_bin, arch, name, version, targets, repo_url, commit_override, out, skip_build,
         engine_name, author, jp_name, use_dispatcher, model_note, arch_desc, bundle, nn_notice):
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

    if nn_notice is None:
        nn_notice = f"eval/nn.bin (c) {author} — 独立したデータであり、エンジン本体 (GPLv3) とは別です。"
    meta = {
        "name": name, "version": version,
        "model": Path(eval_bin).parent.name,
        "engine_name": engine_name or "", "jp_name": jp_name,
        "model_note": model_note, "arch_desc": arch_desc, "nn_notice": nn_notice,
        "repo_url": repo_url, "commit": commit_override or _git_commit(fork), "arch": arch,
    }
    if bundle:
        write_bundle_docs(staging, meta)
    else:
        write_docs(staging, meta)

    zpath = Path(out) / f"{name}-{version}-win.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(staging.parent))
    click.echo(f"packaged -> {zpath}")


if __name__ == "__main__":
    main()
