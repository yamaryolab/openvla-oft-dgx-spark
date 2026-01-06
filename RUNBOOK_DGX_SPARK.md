# DGX Spark（GB10 / aarch64）向け Runbook（Dockerでのセットアップと起動確認）

このRunbookは、DGX Spark上でこのリポジトリ（`openvla-oft`）を **Dockerだけで** 再現可能にし、起動確認（推論/学習/最低限のLIBERO eval）まで完了させる手順です。

## 0. 前提（ホスト側）

1) GPU/ドライバ確認
```bash
nvidia-smi
```

2) Docker/NVIDIA runtime確認
```bash
docker info | sed -n '1,120p'
```

3) NGCログイン（未実施なら）
```bash
docker login nvcr.io
```

## 1. リポジトリ取得

```bash
git clone https://github.com/yamaryolab/openvla-oft-dgx-spark.git
cd openvla-oft-dgx-spark
```

## 2. Dockerイメージをビルド（推奨: LIBERO同梱）

GB10では `nvcr.io/nvidia/pytorch:25.09-py3` 以降が必要です（推奨: `25.12-py3`）。

```bash
docker build -t openvla-oft:dgx \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  --build-arg INSTALL_LIBERO=1 \
  .
```

FlashAttentionも試す場合（失敗しうるので任意）:
```bash
docker build -t openvla-oft:dgx \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  --build-arg INSTALL_LIBERO=1 \
  --build-arg INSTALL_FLASH_ATTN=1 \
  .
```

## 3. 起動（共通：キャッシュ/ログ/データを永続化）

```bash
export OPENVLA_OFT_DIR="$PWD"
export HF_HOME="$HOME/.cache/huggingface"
export RUNS_DIR="$HOME/openvla-oft-runs"
export RLDS_DIR="$HOME/datasets/rlds"
export LIBERO_DATASETS_DIR="$HOME/datasets/libero"

mkdir -p "$HF_HOME" "$RUNS_DIR" "$RLDS_DIR" "$LIBERO_DATASETS_DIR"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  -e MUJOCO_GL=egl \
  -v "$OPENVLA_OFT_DIR:/workspace/openvla-oft" \
  -v "$HF_HOME:/cache/hf" \
  -v "$RUNS_DIR:/runs" \
  -v "$RLDS_DIR:/datasets/rlds" \
  -v "$LIBERO_DATASETS_DIR:/datasets/libero" \
  -e HF_HOME=/cache/hf \
  -e WANDB_DIR=/runs/wandb \
  openvla-oft:dgx bash
```

以降はコンテナ内の操作です（`/workspace/openvla-oft`）。

## 4. 起動確認（依存のimport）

```bash
cd /workspace/openvla-oft

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import tensorflow as tf; import numpy as np; print('tf', tf.__version__, 'numpy', np.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import dlimp; print('dlimp', getattr(dlimp,'__version__','?'))"
python -c "from prismatic.vla.datasets import RLDSDataset; print('RLDSDataset ok')"
python -c "import libero; from libero.libero import get_libero_path; print('LIBERO ok', get_libero_path('bddl_files'))"
```

## 5. LIBERO eval（スモークテスト：短縮実行）

初回はHugging FaceからモデルがDLされるので時間がかかります（`HF_HOME`にキャッシュされます）。

### LIBERO-Spatial（例）
```bash
cd /workspace/openvla-oft
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --use_wandb False
```

### Object / Goal / 10 も同様に（短縮）
```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --num_trials_per_task 1 \
  --use_wandb False

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --num_trials_per_task 1 \
  --use_wandb False

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
  --use_wandb False
```

メモ:
- 終了時にEGLのクリーンアップ警告が出る場合がありますが、完走していれば実害は少ないです。
- 動画はデフォルトで `./rollouts/...` に出ます。ホスト側に残したい場合は `-v "$OPENVLA_OFT_DIR:/workspace/openvla-oft"` を使っているので、そのままホストに残ります。

## 6. RLDS学習（LIBERO例：短縮）

RLDSデータを `/datasets/rlds` に用意した上で、まずは短く回して動作確認します。

```bash
cd /workspace/openvla-oft
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /datasets/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /runs \
  --batch_size 1 \
  --max_steps 1000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only True \
  --use_wandb False
```

