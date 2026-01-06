ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

ARG INSTALL_LIBERO=1
ARG INSTALL_FLASH_ATTN=0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    LIBERO_CONFIG_PATH=/root/.libero \
    PYTHONPATH=/opt/LIBERO

WORKDIR /workspace/openvla-oft

# Global pip constraints:
# - TensorFlow requires NumPy < 2.0 (as of TF 2.20).
# - Some packages (notably newer opencv-python) now depend on NumPy >= 2.
#   We constrain opencv to a NumPy<2-compatible version so RLDS (TF) + LIBERO can coexist.
RUN printf "numpy<2\nopencv-python<4.12\n" > /tmp/pip-constraints.txt

# System deps:
# - build-essential/cmake/ninja: some Python packages may need native extensions
# - ffmpeg: optional, but used by imageio[ffmpeg] in some eval scripts
# - libgl1/libglib2.0-0: common runtime deps for opencv / rendering stacks
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libegl1 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/openvla-oft

# Install this repo + deps.
# Notes:
# - We intentionally do NOT force-install torch/torchvision/torchaudio here; the NGC PyTorch base image provides them.
# - RLDS training requires TensorFlow/TFDS + dlimp.
#   dlimp_openvla pins tensorflow==2.15.0 upstream, which conflicts with Python 3.12; we install dlimp with --no-deps.
# - HF transformers/tokenizers are installed via extra: [hf]. If you need the custom fork, use [hf_openvla_fork].
RUN python -m pip install -U pip setuptools wheel \
    && python -m pip install -e ".[hf]" \
    && python -m pip install -c /tmp/pip-constraints.txt "tensorflow>=2.16" "tensorflow_datasets>=4.9.3" \
    && python -m pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# Optional: FlashAttention (may fail on some aarch64 setups; disabled by default)
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        python -m pip install -e ".[flash_attn]" --no-build-isolation ; \
    fi

# Optional: LIBERO repo + its requirements (for simulation evals; disabled by default if you only need training)
RUN if [ "${INSTALL_LIBERO}" = "1" ]; then \
        git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
        && sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
            /opt/LIBERO/libero/libero/benchmark/__init__.py \
            /opt/LIBERO/libero/lifelong/metric.py \
            /opt/LIBERO/libero/lifelong/evaluate.py \
        && sed -i 's/torch.load(model_path, map_location=map_location)/torch.load(model_path, map_location=map_location, weights_only=False)/g' \
            /opt/LIBERO/libero/lifelong/utils.py \
        && python -m pip install -c /tmp/pip-constraints.txt -e /opt/LIBERO \
        && python -m pip install -c /tmp/pip-constraints.txt -r experiments/robot/libero/libero_requirements.txt \
        && mkdir -p /root/.libero /datasets/libero \
        && printf "benchmark_root: /opt/LIBERO/libero/libero\nbddl_files: /opt/LIBERO/libero/libero/bddl_files\ninit_states: /opt/LIBERO/libero/libero/init_files\ndatasets: /datasets/libero\nassets: /opt/LIBERO/libero/libero/assets\n" > /root/.libero/config.yaml ; \
    fi

# Default to bash for interactive use.
CMD ["bash"]
