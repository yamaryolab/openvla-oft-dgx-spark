# このリポジトリの目的
フォーク元の[moojink/openvla-oft](https://github.com/moojink/openvla-oft)は、2025年現在では古いTensorFlowを使用していたり、環境構築におけるPyTorch系のバージョン指定が極端に厳しかった。
これを、最新のハードウェアであるDGX Sparkでそのまま動かすことは不可能に近いため、Dockerをベースとして依存関係を緩めて再現可能とした本リポジトリを作成した。

## リポジトリ取得
````bash
git clone https://github.com/yamaryolab/openvla-oft-dgx-spark.git
cd openvla-oft-dgx-spark
````
## 前提条件
- GPUドライバやCUDA, Nvidia runtimeなどをインストール
- Docker環境の構築
- [NGC](https://catalog.ngc.nvidia.com/)へのログイン(必要なら)

# 使用手順
## 1. Dockerイメージのビルド
### LIBERO推論環境のみ
````bash
docker build -t openvla-oft:dgx \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  --build-arg INSTALL_LIBERO=1 \
  .
````
ベースイメージ: `nvcr.io/nvidia/pytorch:25.12-py3` (>=25.12-py3ならいい)

### 学習用にFlashAttentionも含む場合（初回時間かかります）:
```bash
docker build -t openvla-oft:dgx \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  --build-arg INSTALL_LIBERO=1 \
  --build-arg INSTALL_FLASH_ATTN=1 \
  .
```

## 2. Dockerコンテナの起動
モデル/データのキャッシュと、学習ログ・RLDSデータをホストへ永続化するためにボリュームをマウントする。
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

## 3. 状態確認（コンテナ内にて）

```bash
cd /workspace/openvla-oft

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import tensorflow as tf; import numpy as np; print('tf', tf.__version__, 'numpy', np.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import dlimp; print('dlimp', getattr(dlimp,'__version__','?'))"
python -c "from prismatic.vla.datasets import RLDSDataset; print('RLDSDataset ok')"
python -c "import libero; from libero.libero import get_libero_path; print('LIBERO ok', get_libero_path('bddl_files'))"
```

### 実行結果例（FlashAttention含む）
````bash
torch 2.10.0a0+b4e4ee81d3.nv25.12 cuda True
tf 2.20.0 numpy 1.26.4
transformers 4.40.1
dlimp ?
Using LIBERO constants:
  NUM_ACTIONS_CHUNK = 8
  ACTION_DIM = 7
  PROPRIO_DIM = 8
  ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
If needed, manually set the correct constants in `prismatic/vla/constants.py`!
RLDSDataset ok
LIBERO ok /opt/LIBERO/libero/libero/bddl_files
````

## 4. [LIBERO eval](/LIBERO.md)の実行

初回はHugging FaceからモデルがDLされるので時間がかかる。（`HF_HOME`にキャッシュされる）

### LIBERO-Spatial（例）
長いので、`--num_trials_per_task`を指定することを推奨。(例：`--num_trials_per_task 1`)
```bash
cd /workspace/openvla-oft

# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10
```

動画はデフォルトで `./rollouts/...` に出る。ホスト側に残したい場合は `-v "$OPENVLA_OFT_DIR:/workspace/openvla-oft"` を使っているので、そのままホストに残る。

## 6. RLDS学習の実行テスト（TensorFlow/TFDSが必要）

RLDSデータを `/datasets/rlds` に用意した上で、まずは短く回して動作確認を行う。

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

# フォーク元リポジトリとの差分
- Dockerベースに完全移行
````bash
#変更したパッケージ制約
sentencepiece: ==0.1.99 → >=0.2.0
timm: ==0.9.10 → >=0.9.10,<1.0.0
torch: ==2.2.0 → 固定ピン撤廃
torchvision: ==0.17.0 → 固定ピン撤廃
torchaudio: ==2.2.0 → 固定ピン撤廃
transformers: git+https://github.com/moojink/transformers-openvla-oft.git → transformers==4.40.1
tensorflow: ==2.15.0 → >=2.16
tensorflow_datasets: ==4.9.3 → >=4.9.3
tensorflow_graphics: ==2021.12.3 → 依存から削除
dlimp: dlimp @ git+...（依存に含める）→ 依存から外し、Dockerで --no-deps インストール
numpy: （明示なし）→ <2 制約を追加
opencv-python: （明示なし）→ <4.12 制約をDockerに追加（TFの numpy<2 と衝突しないようにするため）
flash-attn: （元は手順で ==2.5.5）→ 任意extras化して flash-attn==2.5.5
````

### フォーク元リポジトリのREADME：[README_original.md](/README_original.md)