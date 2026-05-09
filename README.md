# Style-Bert-VITS2 JP-Extra fine-tune on RunPod

One-shot Docker setup that fine-tunes SBV2 JP-Extra on a single-speaker
dataset (the one the irodori-tts dataset UI exports). Designed for ~1–2 h on
an RTX 4090 — total cost typically **$0.50–$1.50** at the published rate
(~$0.39/hr).

## What's in the image

- PyTorch 2.3.1 + CUDA 11.8 (matches SBV2's `torch<2.4` requirement).
- SBV2 cloned at a pinned commit (`66de777e`).
- Base BERT + WavLM + JP-Extra pretrained weights pre-downloaded
  (`initialize.py --skip_default_models`).
- `runpodctl`, `openssh-server`, `tmux`, `ffmpeg` for the in-pod workflow.
- `/opt/run_finetune.py` orchestrator: dataset stage → preprocess → train →
  package into `<model_name>.tar.gz`.

Image size ≈ 8 GB.

## One-time setup

### Image: built and pushed by GitHub Actions

This repo ships with `.github/workflows/build.yml` which builds the Dockerfile
and pushes to **GitHub Container Registry** on every push to `main` (or any
`v*` tag). The first run takes 30–45 min (downloads ~3 GB of HF weights); cached
runs are faster.

After the first successful run, the image is publicly pullable at:

```
ghcr.io/hbar137/sbv2-finetune:latest
```

To create the repo and trigger the first build:

```bash
cd ~/sbv2-finetune
git init && git add . && git commit -m "Initial: SBV2 JP-Extra fine-tune image"
gh repo create hbar137/sbv2-finetune --public --source=. --push
# → workflow starts. Watch with: gh run watch
```

Make the resulting package public once (GitHub defaults new packages to private):

```
GitHub → your profile → Packages → sbv2-finetune → Package settings →
"Change visibility" → Public.
```

After that, RunPod can pull the image without any auth.

### Local machine — `runpodctl` for peer-to-peer file transfer

```bash
curl -L https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
    -o /usr/local/bin/runpodctl && sudo chmod +x /usr/local/bin/runpodctl
```

## Run a fine-tune

### 1. Export your dataset locally

In the irodori-tts UI at <https://tts.eulerai.net/dataset>: review chunks,
hit **Export SBV2**. That writes `~/gozen2ji-dataset/<slug>/export/` with
`wavs/*.wav` + `esd.list`. Then:

```bash
~/irodori-tts/scripts/pack-dataset.sh <slug>
# → produces ~/gozen2ji-dataset/<slug>.zip
```

### 2. Spin up a RunPod GPU pod

- **GPU**: RTX 4090 (24 GB) is the cheapest option that comfortably fits
  SBV2 JP-Extra training at `BATCH_SIZE=4`. Hourly: ~$0.39.
- **Container image**: `ghcr.io/hbar137/sbv2-finetune:latest`
- **Volume**: 20 GB pod volume mounted at `/workspace` (persists for the pod's
  life, more than enough for one fine-tune).
- **Container disk**: 30 GB (image is ~8 GB).
- **Expose**: TCP 22 (SSH), HTTP 8888 (Jupyter optional).
- **Environment variables** (override defaults as needed):

  | name | default | meaning |
  |---|---|---|
  | `MODEL_NAME` | `gozen2ji` | name for output dirs |
  | `EPOCHS` | `100` | epochs (100 is SBV2's recommended fine-tune length) |
  | `BATCH_SIZE` | `4` | per-step batch — 4 fits comfortably on a 4090 |
  | `SAVE_EVERY_STEPS` | `1000` | checkpoint cadence |
  | `USE_JP_EXTRA` | `1` | use JP-Extra (recommended for JA-only voices) |
  | `DATA_ZIP` | `/workspace/data.zip` | where the entrypoint looks for the dataset |
  | `OUTPUT_DIR` | `/workspace/output` | where trained weights land |

  RunPod's `PUBLIC_KEY` is honored: it's added to `/root/.ssh/authorized_keys`.

### 3. Get the dataset onto the pod

After the pod starts, SSH into it via RunPod's web terminal (or `ssh root@<pod>`):

```bash
# from your laptop:
runpodctl send ~/gozen2ji-dataset/<slug>.zip
# → prints a code like "8888-banana-cat-12"

# inside the pod:
cd /workspace
runpodctl receive 8888-banana-cat-12
mv <slug>.zip data.zip
```

### 4. Train

```bash
# inside the pod:
/opt/entrypoint.sh train
```

That runs the orchestrator. Typical timeline on a 4090:

- preprocess (resample → BERT features → style vectors): 5–10 min
- training 100 epochs on ~25 min of audio: 60–90 min

Watch GPU utilization with `nvidia-smi -l 5` in another SSH session.

### 5. Get the trained model off the pod

```bash
# inside the pod, after training completes:
runpodctl send /workspace/output/gozen2ji.tar.gz
# → prints a code

# on your laptop:
runpodctl receive <code>
# → gozen2ji.tar.gz lands in cwd
```

### 6. Stop the pod

Stop it from the RunPod dashboard. **Do not "Terminate"** if you may want to
re-use the volume for further training.

## Cost estimate

| step | wall-clock | cost on RTX 4090 ($0.39/hr) |
|---|---|---|
| pod startup + image pull | ~3 min | $0.02 |
| preprocess | 5–10 min | $0.03–$0.07 |
| 100-epoch fine-tune (25 min audio) | 60–90 min | $0.40–$0.60 |
| package + transfer | <5 min | $0.03 |
| **total per fine-tune** | **~1.5 h** | **~$0.50–$0.80** |

Five experiments fit comfortably under $10.

## Tarball contents

```
gozen2ji.tar.gz
└── gozen2ji/
    ├── G_*.safetensors    # generator (the inference checkpoint)
    ├── D_*.safetensors    # discriminator (only useful for resuming)
    ├── WD_*.safetensors   # WavLM discriminator
    ├── DUR_*.safetensors  # duration predictor
    ├── config.json
    ├── style_vectors.npy
    ├── esd.list
    ├── train.list
    └── val.list
```

For inference you only need `G_<latest>.safetensors`, `config.json`, and
`style_vectors.npy`.

## Troubleshooting

- `pyopenjtalk_dict_for_japanese` errors → `pyopenjtalk-dict` install issue;
  re-run `pip install pyopenjtalk-dict` and try again. Already pre-installed
  in the image.
- GPU OOM → drop `BATCH_SIZE` to `2`. Halves the throughput but trains fine.
- Bad transcripts → re-edit in the dataset UI and re-export — don't try to
  fix transcripts on the pod, the cycle is faster locally.
