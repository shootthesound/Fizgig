# Fizgig on a headless GPU box — RunPod, or any Docker host with an NVIDIA GPU.
#
# Fizgig is a Tkinter desktop app: it needs a display, and there is no web UI to expose. So the
# image runs a virtual screen (Xvfb) with a light window manager on it, and streams that screen
# to a browser over noVNC. You click Connect and get the real app, not a cut-down version.
#
# Deliberately NOT in this image:
#   * Fizgig's source     — cloned/pulled at boot, so a new release reaches users without a
#                           large image rebuild and re-pull.
#   * Model weights       — Klein's repos are gated (BFL require each user to accept the licence
#                           personally, which bundling would bypass), and both families together
#                           are ~80 GB, which takes longer to pull as an image layer than to
#                           fetch from HuggingFace directly. The entrypoint downloads on demand.
#
# No CUDA base image: the torch cu128 wheels bundle their own CUDA runtime, so a plain Ubuntu
# base plus the host's driver is all that's needed, and it keeps the image several GB smaller.

# ---------------------------------------------------------------------------- builder
# Compiles what needs compiling (insightface ships no manylinux wheel) and resolves the whole
# dependency tree into a self-contained venv. None of the toolchain reaches the final image.
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
# The cache purge has to happen in THIS layer: a later RUN would delete the files but the layer
# holding them stays in the image. uv keeps its own wheel cache and ignores PIP_NO_CACHE_DIR,
# which is how a first build shipped 7.8 GB of wheels alongside the 7.8 GB venv they installed.
RUN pip install --upgrade pip uv \
    && uv pip install --link-mode=copy --index-strategy unsafe-best-match \
         -r /tmp/requirements.txt \
    && rm -rf /root/.cache /tmp/requirements.txt

# ---------------------------------------------------------------------------- runtime
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv \
      python3-tk \
      git curl ca-certificates \
      gcc libc6-dev \
      xvfb openbox x11-utils wmctrl \
      libxfont2 libgbm1 libgl1-mesa-dri libegl1 \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      fonts-dejavu-core fonts-liberation fontconfig unzip \
      xdg-utils pcmanfm dbus-x11 \
      openssh-server \
    && rm -rf /var/lib/apt/lists/*
# openssh-server sits here unused unless PUBLIC_KEY is set at boot (see entrypoint.sh) — the
# file manager + runpodctl remain the default, keyless transfer path. This is for anyone who
# wants a real shell for debugging: RunPod's own "SSH Terminal Access" toggle sets PUBLIC_KEY
# for you, so opting in there wires this up with no template changes; anywhere else, set
# PUBLIC_KEY to your own public key. Off by default: no open port 22 for someone who never asked.
# gcc + libc6-dev are RUNTIME dependencies, not builder leftovers: torch.compile has
# inductor/triton build small host-side C stubs while training runs, and without a compiler
# every compiled run died with "Failed to find C compiler". The app now declines compile
# cleanly when no cc exists, but shipping one is what makes compile actually WORK on pods —
# it's a real speed win on the INT8/NF4 paths. Not build-essential: g++/make/dpkg-dev would
# triple the size for nothing.
# pcmanfm is what makes the "Open folder" buttons work: _open_in_file_manager shells out to
# xdg-open, which needs something to hand the folder to.
#
# dbus-x11 is not optional decoration. Firefox hands a URL to an already-running instance over the
# session bus; with no bus, the second webbrowser.open() of a session starts a SECOND firefox,
# hits the profile lock, and puts up a "Close Firefox" error instead of the page. The first link
# you click works and every one after it fails — which is exactly the kind of bug that looks like
# it works when you only test it once.

# A browser, because without one every "Get help on YouTube", "Buy me a coffee" and Download link
# in Fizgig calls webbrowser.open() into the void — the button depresses and nothing happens, with
# no error. It also fixes the things that are NOT external links and could never be worked around
# client-side: the samples gallery serves on 127.0.0.1 inside this container (likeness scoring and
# the Run Visualiser live there), and the Profiler and Repair Studio open local HTML reports.
#
# NOT `apt-get install firefox`: on Ubuntu 22.04 that is a snap transition stub, and snaps need
# systemd so they cannot run in a container. It would build cleanly and fail silently at runtime —
# precisely the bug being fixed here. Same for chromium-browser. Mozilla publishes a real .deb
# repo for this, and the APT pin is load-bearing: without it Ubuntu's stub outranks the Mozilla
# package and the snap wins anyway.
RUN install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://packages.mozilla.org/apt/repo-signing-key.gpg \
         -o /etc/apt/keyrings/packages.mozilla.org.asc \
    && echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" \
         > /etc/apt/sources.list.d/mozilla.list \
    && printf 'Package: *\nPin: origin packages.mozilla.org\nPin-Priority: 1000\n' \
         > /etc/apt/preferences.d/mozilla \
    && apt-get update && apt-get install -y --no-install-recommends firefox \
    && rm -rf /var/lib/apt/lists/* \
    && firefox --version
# Firefox runs as root here (the container has no other user) and its content sandbox objects to
# that; disabling it is safe in a single-user throwaway container and is the difference between
# the browser opening and silently dying.
ENV MOZ_DISABLE_CONTENT_SANDBOX=1

# Getting datasets in and LoRAs out — the two things that make a rented pod usable at all.
#
# runpodctl is RunPod's own transfer CLI (one-time codes, no ports to open, no SSH keys). It ships
# preinstalled on RunPod's base images but NOT on a custom one like this, so users would otherwise
# lose the transfer method the RunPod docs tell them to use.
#
# filebrowser is the GUI answer, and the one that matters for Fizgig's audience: a browser tab you
# drag a dataset folder into and download finished LoRAs from. Everything else on offer — runpodctl,
# scp, rsync — is a terminal, and the entire point of this image is that people get a GUI instead.
RUN curl -fsSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
      -o /usr/local/bin/runpodctl \
    && chmod +x /usr/local/bin/runpodctl \
    && runpodctl version \
    && curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash \
    && filebrowser version

# Fizgig's whole layout is measured in Segoe UI, which does not exist on Linux — Tk silently
# substitutes DejaVu Sans, which renders the same string ~20% wider (321px vs 268px measured).
# That is not cosmetic: every wraplength, column width and window geometry in the app is tuned to
# the narrower metrics, which is why the tab strip truncated "Preferences" before this.
#
# Selawik is Microsoft's own metric-compatible open substitute for Segoe UI (SIL OFL), so aliasing
# it fixes the measurements everywhere at once with no changes to Fizgig itself.
RUN curl -fsSL https://github.com/microsoft/Selawik/releases/download/1.01/Selawik_Release.zip \
      -o /tmp/selawik.zip \
    && mkdir -p /usr/share/fonts/truetype/selawik \
    && unzip -j -o /tmp/selawik.zip '*.ttf' -d /usr/share/fonts/truetype/selawik \
    && rm /tmp/selawik.zip \
    && printf '%s\n' \
      '<?xml version="1.0"?>' \
      '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">' \
      '<fontconfig>' \
      '  <alias binding="same">' \
      '    <family>Segoe UI</family>' \
      '    <accept><family>Selawik</family></accept>' \
      '  </alias>' \
      '</fontconfig>' \
      > /etc/fonts/conf.d/99-segoe-selawik.conf \
    && fc-cache -f
# python3-tk is the one that is easy to miss and fatal: without it Tkinter is absent and the GUI
# cannot start at all. The libgl/libglib set is what opencv needs on a headless box.
# build-essential stays out — the full toolchain was only ever needed to compile insightface in
# the builder; the runtime carries just gcc for torch.compile (see above).

COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

# Everything the user creates lives here so a RunPod network volume keeps it across pods:
# models, datasets, output LoRAs, latent cache, and the HuggingFace/InsightFace caches — which
# would otherwise re-download Florence-2 and the face model on every single boot.
# Stamped by CI from the release tag. The app shows it next to the app's own commit, because the
# two deliberately diverge: the image is pinned in the RunPod template while Fizgig itself updates
# from git at every pod start. Without this, "which image are you on?" is unanswerable, which is
# exactly the failure mode that makes a mutable :latest tag risky.
ARG FIZGIG_IMAGE_VERSION=dev
ENV FIZGIG_IMAGE_VERSION=${FIZGIG_IMAGE_VERSION}

# FIZGIG_POD is what the app keys off to show its RunPod controls instead of the "run on a rented
# GPU" advert. Our own marker rather than RUNPOD_POD_ID: a provider's variable names are theirs
# to change, and another host would set different ones entirely.
ENV FIZGIG_POD=1 \
    FIZGIG_HOME=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    INSIGHTFACE_HOME=/workspace/.insightface \
    XDG_CACHE_HOME=/workspace/.cache
WORKDIR /workspace

# KasmVNC replaces Xvfb + x11vnc + websockify + noVNC with one server.
#
# The old stack was laggy over the internet and there was no tuning left to do client-side: noVNC's
# defaults are LAN-shaped, and even at compression 9 it was still shipping raw-ish framebuffer
# updates. KasmVNC encodes in WebP (~30% smaller than the JPEG/Tight the old stack managed),
# encodes multi-threaded, and mixes WebP/JPEG automatically based on spare CPU — of which a
# training pod has plenty. It also serves its own web client, so websockify and the noVNC package
# both disappear.
#
# The other win is dynamic resize: the desktop follows the browser window, so SCREEN_SIZE stops
# being a guess between "fits a laptop" and "less scrolling" and just becomes whatever the user's
# window is.
RUN curl -fsSL https://github.com/kasmtech/KasmVNC/releases/download/v1.5.0/kasmvncserver_jammy_1.5.0_amd64.deb \
      -o /tmp/kasmvnc.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/kasmvnc.deb \
    && rm /tmp/kasmvnc.deb && rm -rf /var/lib/apt/lists/* \
    && Xvnc -version 2>&1 | head -1

# HTTP not HTTPS: RunPod (and any sane host) terminates TLS at its own proxy, and leaving
# require_ssl on means the proxy talks plaintext to an endpoint expecting TLS and the connection
# just fails. allow_resize is what makes the desktop track the browser window.
RUN printf '%s\n' \
      'desktop:' \
      '  resolution:' \
      '    width: 1600' \
      '    height: 1400' \
      '  allow_resize: true' \
      '  pixel_depth: 24' \
      'network:' \
      '  protocol: http' \
      '  interface: 0.0.0.0' \
      '  websocket_port: 6080' \
      '  ssl:' \
      '    require_ssl: false' \
      'user_session:' \
      '  new_session_disconnects_existing_exclusive_session: true' \
      '  concurrent_connections_prompt: false' \
      '  idle_timeout: never' \
      > /etc/kasmvnc/kasmvnc.yaml

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# noVNC over HTTP — the only port that needs exposing. 5900 stays internal on purpose: raw VNC
# is unencrypted, and the host's HTTPS proxy is what makes this safe to reach over the internet.
# 8081 is reserved for the app to serve on (a phone-facing run monitor). 22 only ever listens
# if PUBLIC_KEY is set (see entrypoint.sh) — declaring it here is documentation, same as 8081.
# Declaring these is documentation — RunPod exposes what the TEMPLATE lists — but it keeps the
# ports in one place and works for a plain `docker run -p`.
EXPOSE 6080 8080 8081 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
