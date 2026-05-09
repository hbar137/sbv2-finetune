#!/usr/bin/env python
"""SBV2 JP-Extra fine-tune orchestrator.

Reads dataset from $DATA_DIR (or $DATA_ZIP), runs preprocess_all + training,
copies outputs to $OUTPUT_DIR, packages as <model>.tar.gz for easy transfer.

Expected dataset layout (one of):
  $DATA_ZIP                                 (zip file with esd.list + wavs/)
  $DATA_DIR/wavs/*.wav   $DATA_DIR/esd.list
  $DATA_DIR/raw/*.wav    $DATA_DIR/esd.list   (already-SBV2-shaped)

esd.list format:  <wav_filename>|<speaker>|JP|<transcript>
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

SBV2_ROOT = Path("/opt/Style-Bert-VITS2")


def env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[run_finetune] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def stage_dataset(src_dir: Path, src_zip: Path, dest_dir: Path) -> None:
    """Copy/extract dataset into SBV2's expected Data/<model>/ layout."""
    raw = dest_dir / "raw"
    esd = dest_dir / "esd.list"
    raw.mkdir(parents=True, exist_ok=True)

    if src_zip.exists():
        print(f"[run_finetune] extracting {src_zip}", flush=True)
        scratch = dest_dir / "_unpack"
        scratch.mkdir(exist_ok=True)
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(scratch)
        src_dir = scratch  # then fall through to copy logic below

    if not src_dir.exists():
        sys.exit(f"[run_finetune] no dataset found at {src_dir} or {src_zip}")

    # find esd.list (root or one level down)
    esd_src = next(iter(list(src_dir.glob("esd.list")) + list(src_dir.glob("*/esd.list"))), None)
    if esd_src is None:
        sys.exit(f"[run_finetune] esd.list not found under {src_dir}")
    shutil.copy2(esd_src, esd)

    # find wavs (wavs/, raw/, or root)
    wav_src_dir = None
    for cand in (esd_src.parent / "wavs", esd_src.parent / "raw", esd_src.parent):
        if cand.exists() and any(cand.glob("*.wav")):
            wav_src_dir = cand
            break
    if wav_src_dir is None:
        sys.exit(f"[run_finetune] no .wav files found near {esd_src}")
    n = 0
    for w in wav_src_dir.glob("*.wav"):
        shutil.copy2(w, raw / w.name)
        n += 1
    print(f"[run_finetune] staged {n} wavs + esd.list under {dest_dir}", flush=True)


def preprocess(model_name: str, batch_size: int, epochs: int, save_every_steps: int,
               use_jp_extra: bool) -> None:
    """Call SBV2's preprocess_all directly via Python import."""
    sys.path.insert(0, str(SBV2_ROOT))
    os.chdir(SBV2_ROOT)
    from gradio_tabs.train import preprocess_all
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker

    pyopenjtalk_worker.initialize_worker()
    print(f"[run_finetune] preprocess_all(model={model_name}, bs={batch_size}, "
          f"ep={epochs}, save_every={save_every_steps}, jp_extra={use_jp_extra})", flush=True)
    preprocess_all(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        num_processes=2,
        normalize=False,
        trim=False,
        freeze_EN_bert=False,
        freeze_JP_bert=False,
        freeze_ZH_bert=False,
        freeze_style=False,
        freeze_decoder=False,
        use_jp_extra=use_jp_extra,
        val_per_lang=0,
        log_interval=200,
        yomi_error="skip",
    )


def train(model_name: str, use_jp_extra: bool, assets_root: Path) -> None:
    dataset_path = SBV2_ROOT / "Data" / model_name
    config_path = dataset_path / "config.json"
    train_list = dataset_path / "train.list"
    if not train_list.exists() or train_list.stat().st_size == 0:
        sys.exit(f"[run_finetune] preprocess produced no train.list at {train_list} — "
                 "check earlier log for the actual subprocess error (SBV2's "
                 "preprocess_all logs subprocess failures but does not raise).")
    script = "train_ms_jp_extra.py" if use_jp_extra else "train_ms.py"
    run([
        "python", script,
        "--config", str(config_path),
        "--model", str(dataset_path),
        "--assets_root", str(assets_root),
    ], cwd=SBV2_ROOT)


def collect_outputs(model_name: str, assets_root: Path, output_dir: Path) -> Path:
    """Copy trained model + config to OUTPUT_DIR, then tar it for easy transfer.

    SBV2 writes two parallel sets of checkpoints during training:
      Data/<model>/models/{G,D,WD}_<step>.pth  — generator + discriminators with
        optimizer state (needed to resume training; SBV2 rotates older ones out
        so only the latest set survives at end-of-run).
      model_assets/<model>/<model>_e<ep>_s<step>.safetensors  — generator-only
        weights for every save_every_steps (needed for inference; all kept).

    We bundle both: the .safetensors give you a menu of inference checkpoints
    to A/B between, the .pth lets you resume training from where this run
    stopped if you want to extend epochs.
    """
    dataset_path = SBV2_ROOT / "Data" / model_name
    models_dir = dataset_path / "models"
    out = output_dir / model_name
    out.mkdir(parents=True, exist_ok=True)

    # Resume-training state: latest G/D/WD .pth (with optimizer) — SBV2 keeps
    # only the most recent set, so this is just the final checkpoint pair.
    for src in models_dir.glob("*.pth"):
        shutil.copy2(src, out / src.name)
    # Belt-and-suspenders for any .safetensors that happen to live here too
    # (older SBV2 versions, or future variants).
    for src in models_dir.glob("*.safetensors"):
        shutil.copy2(src, out / src.name)

    for fname in ("config.json", "esd.list", "style_vectors.npy", "train.list", "val.list"):
        src = dataset_path / fname
        if src.exists():
            shutil.copy2(src, out / fname)

    # Per-epoch generator safetensors for inference (one per SAVE_EVERY_STEPS).
    assets_export = assets_root / model_name
    if assets_export.exists():
        for f in assets_export.iterdir():
            if f.is_file():
                shutil.copy2(f, out / f.name)

    tar_path = output_dir / f"{model_name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out, arcname=model_name)
    print(f"[run_finetune] packaged {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return tar_path


def main() -> None:
    model_name = env("MODEL_NAME", "gozen2ji")
    epochs = int(env("EPOCHS", "100"))
    batch_size = int(env("BATCH_SIZE", "4"))
    save_every_steps = int(env("SAVE_EVERY_STEPS", "1000"))
    use_jp_extra = env("USE_JP_EXTRA", "1") == "1"
    data_dir = Path(env("DATA_DIR", "/workspace/data"))
    data_zip = Path(env("DATA_ZIP", "/workspace/data.zip"))
    output_dir = Path(env("OUTPUT_DIR", "/workspace/output"))
    assets_root = Path(env("ASSETS_ROOT", str(SBV2_ROOT / "model_assets")))

    output_dir.mkdir(parents=True, exist_ok=True)
    sbv2_dataset_path = SBV2_ROOT / "Data" / model_name
    sbv2_dataset_path.mkdir(parents=True, exist_ok=True)

    stage_dataset(data_dir, data_zip, sbv2_dataset_path)
    preprocess(model_name, batch_size, epochs, save_every_steps, use_jp_extra)
    train(model_name, use_jp_extra, assets_root)
    tar_path = collect_outputs(model_name, assets_root, output_dir)

    print(f"\n[run_finetune] DONE.\n  outputs: {output_dir}/{model_name}\n  archive: {tar_path}",
          flush=True)
    print("[run_finetune] Transfer the archive off the pod with:", flush=True)
    print(f"    runpodctl send {tar_path}", flush=True)


if __name__ == "__main__":
    main()
