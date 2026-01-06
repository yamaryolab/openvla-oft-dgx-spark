# Setup Instructions

## Docker (推奨: DGX Spark / aarch64 / 新しめGPU)

このリポジトリは学習時にRLDS（TensorFlow/TFDS + `dlimp`）を使います。一方で、DGX Spark（GB10）のような新しめGPU/`aarch64`環境では、
従来の「conda + 固定バージョンのPyTorch/TensorFlow」をそのまま再現するのが難しいため、NVIDIA NGCのPyTorchコンテナをベースにする方法を推奨します。

DGX Spark向けに「最初から最後まで（ビルド→起動→動作確認）」の手順を `RUNBOOK_DGX_SPARK.md` にまとめています。

### 前提

- NVIDIA Driver / CUDA がインストール済み（ホスト側で `nvidia-smi` が動く）
- Docker / NVIDIA Container Toolkit が有効（`docker info` で `Default Runtime: nvidia` などが確認できる）
- NGCへログイン済み（必要なら `docker login nvcr.io`）

### 1) Dockerイメージをビルド

DGX Spark（GB10）では、少なくとも `nvcr.io/nvidia/pytorch:25.09-py3` 以降（推奨: `25.12-py3`）を使ってください。

```bash
docker build -t openvla-oft:dgx \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  .
```

`Dockerfile` はデフォルトで `pip install -e ".[hf]"`（Hugging Face Transformers）に加え、RLDS学習に必要な
`tensorflow/tensorflow_datasets` と `dlimp`（`--no-deps` で導入）をインストールします。
もしOpenVLA用のカスタムtransformersフォークを使いたい場合は、`Dockerfile` の `.[hf]` を `.[hf_openvla_fork]` に切り替えてください。

LIBEROシミュレーション評価もコンテナ内で完結させたい場合は、LIBEROもビルド時に同梱できます（推奨: ON）:

```bash
docker build -t openvla-oft:dgx \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  --build-arg INSTALL_LIBERO=1 \
  .
```

FlashAttentionは `aarch64` では失敗する場合があるためデフォルトOFFです。必要であれば:

```bash
docker build -t openvla-oft:dgx \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  --build-arg INSTALL_FLASH_ATTN=1 \
  .
```

### 2) コンテナ起動（共通）

モデル/データのキャッシュと、学習ログ・RLDSデータをホストへ永続化するためにボリュームをマウントします。

```bash
export OPENVLA_OFT_DIR="$PWD"
export HF_HOME="$HOME/.cache/huggingface"
export RUNS_DIR="$HOME/openvla-oft-runs"
export RLDS_DIR="$HOME/datasets/rlds"

mkdir -p "$HF_HOME" "$RUNS_DIR" "$RLDS_DIR"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  -v "$OPENVLA_OFT_DIR:/workspace/openvla-oft" \
  -v "$HF_HOME:/cache/hf" \
  -v "$RUNS_DIR:/runs" \
  -v "$RLDS_DIR:/datasets/rlds" \
  -e HF_HOME=/cache/hf \
  -e TRANSFORMERS_CACHE=/cache/hf/transformers \
  -e WANDB_DIR=/runs/wandb \
  openvla-oft:dgx bash
```

以降のコマンドはコンテナ内で `/workspace/openvla-oft` にいる前提です。

### 3) 推論（TFなしで動作）

```bash
cd /workspace/openvla-oft
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python vla-scripts/extern/verify_openvla.py  # 任意（チェック用）
```

### 4) 学習（RLDS: TensorFlow/TFDSが必要）

RLDSデータを用いた学習は `vla-scripts/finetune.py` を使います。例（LIBERO）:

```bash
cd /workspace/openvla-oft
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /datasets/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /runs \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 1000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank 32
```

ALOHA実機の学習も最終的には同じ `finetune.py` を使いますが、まずはLIBEROのRLDSで学習パスを安定させ、その後にALOHAのRLDSデータ作成（別リポジトリのデータビルダー）へ進むのが安全です（詳細は `ALOHA.md` を参照）。

### 5) FlashAttention（任意）

`flash-attn` は `aarch64` 環境では導入が難しい場合があります。まずは無しで学習を通し、必要であれば追加してください。

```bash
cd /workspace/openvla-oft
python -m pip install -U pip
python -m pip install -e ".[flash_attn]" --no-build-isolation
```

## Conda（非推奨: 互換性問題が出やすい）

従来通りcondaでの構築も可能ですが、DGX Spark（GB10）/`aarch64`/新しめCUDAでは、TensorFlow・`tensorflow_graphics`・`flash-attn`などの
ビルド/ホイール互換で詰まりやすいため、まずはDockerの手順を推奨します。
