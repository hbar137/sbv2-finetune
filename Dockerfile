# Style-Bert-VITS2 JP-Extra fine-tune image, RunPod-friendly.
# Base: official PyTorch CUDA 11.8 image (matches SBV2's torch<2.4 requirement).
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# OS deps. ffmpeg for resample, openssh-server so RunPod web SSH works,
# tmux/curl/unzip for the in-pod workflow, git for the SBV2 clone.
# pkg-config + libav* headers are required by PyAV which faster-whisper
# (pinned by SBV2 to 0.10.1) builds from sdist.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates unzip zip tmux \
        openssh-server build-essential \
        pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

# runpodctl for peer-to-peer file transfer in/out of the pod.
RUN curl -L https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
        -o /usr/local/bin/runpodctl \
    && chmod +x /usr/local/bin/runpodctl

# Clone SBV2 (pinned).
ARG SBV2_REF=66de777e06392c0f313600be03c43ef96658b244
RUN git clone https://github.com/litagin02/Style-Bert-VITS2.git /opt/Style-Bert-VITS2 \
    && cd /opt/Style-Bert-VITS2 && git checkout "${SBV2_REF}"

WORKDIR /opt/Style-Bert-VITS2

# torch / torchaudio already in base image; install the rest. SBV2 pins
# torch<2.4 in requirements, so don't reinstall torch — just everything else.
RUN sed -i '/^torch/d; /^torchaudio/d' requirements.txt \
    && pip install -r requirements.txt

# Pre-download base weights so training starts immediately. --skip_default_models
# skips the demo voice models (~3 GB) we don't need for fine-tuning.
RUN python initialize.py --skip_default_models

# Pre-download the Whisper model used by transcribe.py (we don't run it here,
# but having it cached lets the user re-transcribe on the pod if needed).
# Skipped — saves ~3 GB.

COPY run_finetune.py /opt/run_finetune.py
COPY entrypoint.sh   /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Defaults (override via -e on docker run / RunPod env vars).
ENV MODEL_NAME=gozen2ji \
    EPOCHS=100 \
    BATCH_SIZE=4 \
    SAVE_EVERY_STEPS=1000 \
    USE_JP_EXTRA=1 \
    DATA_DIR=/workspace/data \
    OUTPUT_DIR=/workspace/output \
    DATA_ZIP=/workspace/data.zip

EXPOSE 22 8888

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["wait"]
