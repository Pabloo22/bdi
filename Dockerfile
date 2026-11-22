# Image for rented GPU machines (docs/e0.md, "Running on vast.ai"): Python 3.12, Poetry and the
# whole `gpu` group already installed. Build and push with scripts/build_image.sh.
#
# The `base` tag, not `runtime` or `devel`: torch and vllm bring their own CUDA
# libraries as nvidia-*-cu12 wheels, so the only thing needed from the image is
# NVIDIA_VISIBLE_DEVICES/NVIDIA_DRIVER_CAPABILITIES, which is what exposes the
# host driver under vast.ai's container runtime. 12.4 matches torch 2.6's cu124
# wheels and the "CUDA 12.4 or newer" driver filter in docs/e0.md. If some wheel
# turns out to dlopen a system library after all, 12.4.1-runtime-ubuntu22.04 adds
# cuBLAS and NCCL back for ~2 GB.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# python3.12 comes from deadsnakes because 22.04 ships 3.10 and pyproject pins
# >=3.12,<3.13. build-essential and python3.12-dev are needed at run time, not
# only at install: Triton (under torch and vLLM) compiles a small C launcher
# against Python.h the first time it runs a kernel. git is for cloning the repo
# on the box; rsync is the remote end of the "bringing results home" command in
# docs/e0.md.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential git rsync && \
    rm -rf /var/lib/apt/lists/*

# Deliberately no `update-alternatives` for python3: on this base /usr/bin/python3
# is the 3.10 that apt's own tooling (add-apt-repository) imports, and repointing
# it breaks them. The project venv goes on PATH below instead, so `python` is
# 3.12 for anything a shell on the box actually runs.

# Same Poetry version that wrote poetry.lock, in its own venv so its dependencies
# never mix with the project's; only the `poetry` command goes on PATH. From PyPI
# with pip rather than install.python-poetry.org's `curl | python3 -`: the install
# below needs PyPI anyway, and that site can be unreachable while PyPI is not. When
# it was, curl's failure never reached the shell (a pipe reports only its last
# command), so python3 ran an empty script and the step "succeeded" without Poetry.
RUN python3.12 -m venv /opt/poetry && \
    /opt/poetry/bin/pip install --no-cache-dir poetry==2.2.1 && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Resolve and install the dependency set at build time rather than on the rental
# box. Two reasons: `poetry install --with gpu` on a fresh instance downloads and
# unpacks ~15 GB (torch + the nvidia-*-cu12 wheels + vLLM + flash-attn) before any
# GPU work starts, and doing it here means every box starts from the bytes this
# image was tested with. Only the manifests and the vendored verl are copied
# (.dockerignore keeps everything else out of the build), so editing bdi/ never
# invalidates this layer. verl has to come along because the gpu group installs
# it from its folder, so it goes at the same relative path pyproject names.
WORKDIR /opt/app
COPY pyproject.toml poetry.lock ./
COPY bdi/vendored/oel/verl bdi/vendored/oel/verl
# in-project is load-bearing, not cosmetic: without it poetry puts the venv under
# /root/.cache/pypoetry/virtualenvs/ next to its download cache, so the cleanup
# below would delete the environment it just built. --no-root because the bdi
# package itself is installed from the clone on the box (see below).
RUN poetry config virtualenvs.in-project true && \
    poetry env use /usr/bin/python3.12 && \
    poetry install --no-root --with gpu --no-interaction && \
    rm -rf /root/.cache/pypoetry /root/.cache/pip

ENV VIRTUAL_ENV=/opt/app/.venv
ENV PATH="/opt/app/.venv/bin:$PATH"

# With the venv pre-activated and creation disabled, poetry inside the cloned
# repo reuses /opt/app/.venv instead of building a second one in-project. That
# keeps docs/e0.md's setup working verbatim, and its `poetry install --with gpu`
# takes seconds: every locked package is already here, so it only installs the
# bdi package (the tests import it) and reinstalls verl from the clone's own
# vendored copy. If the clone's poetry.lock is newer than this image, the same
# command downloads just the difference. Set only after the install above, which
# needs creation enabled.
ENV POETRY_VIRTUALENVS_CREATE=false

# Keep the ~15 GB Qwen2.5-7B-Instruct download next to the code on the big rental
# disk instead of in /root/.cache, so it is one directory to point at a volume or
# clear when reclaiming space.
ENV HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace
