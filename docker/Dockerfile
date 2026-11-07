# Fizgig on a headless GPU box — RunPod, or any Docker host with an NVIDIA GPU.
#
# Fizgig is a web app here: the pod serves its web UI on port 8081 and you use it in your own browser, from any
# device. No virtual screen, remote desktop or browser inside the pod - which is what makes this image small and quick
# to start.
#
# Deliberately NOT in this image:
#   * Fizgig's source     — cloned/pulled at boot, so a new release reaches users without a
#                           large image rebuild and re-pull.
#   * Model weights       — Klein's repos are gated (BFL require each user to accept the licence
#                           personally, which bundling would bypass), and the families together
#                           are far larger than it is worth pulling as an image layer. Preferences >
#                           Download models fetches them on demand.
#
# No CUDA base image: the torch cu134 wheels bundle their own CUDA runtime, so a plain Ubuntu
# base plus the host's driver is all that's needed, and it keeps the image several GB smaller.

# ---------------------------------------------------------------------------- builder
# Compiles what needs compiling (insightface ships no manylinux wheel) and resolves the whole
# dependency tree into a self-contained venv. None of the toolchain reaches the final image.
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
COPY uv_install_deps.py /tmp/uv_install_deps.py
# The cache purge has to happen in THIS layer: a later RUN would delete the files but the layer
# holding them stays in the image. uv keeps its own wheel cache and ignores PIP_NO_CACHE_DIR,
# which is how a first build shipped 7.8 GB of wheels alongside the 7.8 GB venv they installed.
RUN pip install --upgrade pip uv \
    && python3 /tmp/uv_install_deps.py /tmp/requirements.txt "$VIRTUAL_ENV" \
    && rm -rf /root/.cache /tmp/requirements.txt /tmp/uv_install_deps.py

# ---------------------------------------------------------------------------- runtime
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv \
      git curl ca-certificates \
      gcc libc6-dev python3-dev \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      fonts-dejavu-core fonts-liberation fontconfig \
      openssh-server \
    && rm -rf /var/lib/apt/lists/*
# openssh-server sits here unused unless PUBLIC_KEY is set at boot (see entrypoint.sh) — the
# web UI's uploads, the file manager and runpodctl remain the keyless transfer paths. RunPod's own
# "SSH Terminal Access" toggle sets PUBLIC_KEY for you; anywhere else, set it to your own public key.
# gcc + libc6-dev are RUNTIME dependencies, not builder leftovers: torch.compile has
# inductor/triton build small host-side C stubs while training runs, and without a compiler
# every compiled run died with "Failed to find C compiler". Not build-essential: g++/make/dpkg-dev
# would triple the size for nothing.
# python3-dev rides with them for the same reason: triton's stubs #include <Python.h>, so a
# compiler without the headers gets further and then dies with "Python.h: No such file or
# directory" (reported by a RunPod user on Krea 2, 7 Aug 2026).
# The libgl / libglib / libsm / libxext / libxrender set is what opencv needs on a headless box; the fonts are
# what the pictures Fizgig draws text on (sweep strips, comparison sheets, clip labels) find.

# Getting datasets in and LoRAs out besides the web UI's own uploads and downloads.
#
# runpodctl is RunPod's own transfer CLI (one-time codes, no ports to open, no SSH keys). It ships
# preinstalled on RunPod's base images but NOT on a custom one like this.
#
# filebrowser: a browser tab on port 8080 for dragging whole folders in and out of /workspace.
RUN curl -fsSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
      -o /usr/local/bin/runpodctl \
    && chmod +x /usr/local/bin/runpodctl \
    && runpodctl version \
    && curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash \
    && filebrowser version

COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

# Stamped by CI from the release tag. The app shows it next to the app's own commit, because the
# two deliberately diverge: the image is pinned in the RunPod template while Fizgig itself updates
# from git at every pod start. Without this, "which image are you on?" is unanswerable.
ARG FIZGIG_IMAGE_VERSION=dev
ENV FIZGIG_IMAGE_VERSION=${FIZGIG_IMAGE_VERSION}

# Everything the user creates lives under /workspace so a RunPod volume keeps it across pods: models,
# datasets, output LoRAs, latent cache, and the HuggingFace/InsightFace caches — which would otherwise
# re-download the captioner and the face model on every single boot.
# FIZGIG_POD is what the app keys off to show its RunPod settings (stop the pod when a run finishes).
ENV FIZGIG_POD=1 \
    FIZGIG_HOME=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    INSIGHTFACE_HOME=/workspace/.insightface \
    XDG_CACHE_HOME=/workspace/.cache
WORKDIR /workspace

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 8081 the web UI, 8080 the file manager, 22 only if PUBLIC_KEY is set (see entrypoint.sh). Declaring them is
# documentation - RunPod exposes what the TEMPLATE lists - but it keeps the ports in one place and works for a plain
# `docker run -p`.
EXPOSE 8081 8080 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
