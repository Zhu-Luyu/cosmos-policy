#!/bin/bash
#SBATCH --job-name=cosmos_array      # 阵列作业名称
#SBATCH --partition=i64m1tga40u      # 使用高性价比的 A40 队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-23                 # 【核心魔法】告诉 SLURM 启动 24 个子任务 (索引 0 到 23)
#SBATCH --output=logs_%A_%a.out      # %A 是主作业ID，%a 是子任务的阵列ID
#SBATCH --error=logs_%A_%a.err
#SBATCH --time=24:00:00

# 1. 激活环境
eval "$(conda shell.bash hook)"
conda activate cosmos-env
cd ~/cosmos-policy

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HOME/models"
export TRANSFORMERS_CACHE="$HOME/models"

export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export PATH="$HOME/cosmos-policy/bin:$PATH"

# 2. 定义 24 个 RoboCasa 任务名称的数组
# (⚠️请核对你数据集里的实际任务名称，以下为 RoboCasa 常见任务列表)
TASKS=(
    # 门操作任务 (6个)
    "OpenSingleDoor" "CloseSingleDoor" "OpenDoubleDoor" "CloseDoubleDoor" "OpenDrawer" "CloseDrawer"
    
    # 拾取和放置任务 (8个)
    "PnPCounterToCab" "PnPCabToCounter" "PnPCounterToSink" "PnPSinkToCounter" 
    "PnPCounterToStove" "PnPStoveToCounter" "PnPCounterToMicrowave" "PnPMicrowaveToCounter"
    
    # 电器控制任务 (7个)
    "TurnOnMicrowave" "TurnOffMicrowave" "TurnOnSinkFaucet" "TurnOffSinkFaucet" 
    "TurnSinkSpout" "TurnOnStove" "TurnOffStove"
    
    # 咖啡制作任务 (3个)
    "CoffeeSetupMug" "CoffeeServeMug" "CoffeePressButton"
)

# 3. 获取当前子任务分配到的具体任务名称
# $SLURM_ARRAY_TASK_ID 就是 0 到 23 之间的数字
CURRENT_TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "=================================================="
echo "🚀 开始在 GPU 上评估任务: $CURRENT_TASK"
echo "=================================================="

# 4. 运行评估 (指定当前任务，并设为 50 次)
uv run --no-sync python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path $HOME/models/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path $HOME/models/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path $HOME/models/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name $CURRENT_TASK \
    --num_trials_per_task 50 \
    --run_id_note chkpt45000--5stepAct--seed195--deterministic \
    --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_collection False