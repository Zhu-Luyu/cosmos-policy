#!/bin/bash
#SBATCH --job-name=cosmos_eval       # 作业名称为 cosmos_eval [cite: 2290, 2291]
#SBATCH --partition=i64m1tga40u     # 指定计算分区，这里使用正式的 A800 GPU 队列 [cite: 2292, 2293, 2448]
#SBATCH --nodes=1                    # 请求 1 个计算节点 [cite: 2298, 2299]
#SBATCH --ntasks-per-node=1          # 每个节点运行 1 个任务 [cite: 2300]
#SBATCH --cpus-per-task=16           # 为该任务分配 16 个 CPU 核心 [cite: 2301]
#SBATCH --gres=gpu:1                 # 申请 1 张 GPU 卡 [cite: 2302, 2305]
#SBATCH --output=%j.out              # 将标准输出日志保存到当前目录，%j 会被替换为真实作业 ID 
#SBATCH --error=%j.err               # 将报错输出日志保存到当前目录 [cite: 2296, 2297]
#SBATCH --time=24:00:00              # 申请预计运行时间 (最多 24 小时)

# 1. 激活环境
eval "$(conda shell.bash hook)"
conda activate cosmos-env
cd ~/cosmos-policy

# 2. 确保环境变量指向国内镜像和你的本地模型目录
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HOME/models"
export TRANSFORMERS_CACHE="$HOME/models"

# 3. 拦截系统的 ldconfig，解决 HPC 无 root 权限读取动态库的 bug
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export PATH="$HOME/cosmos-policy/bin:$PATH"

# 4. 运行完整评估脚本 (50 trials)
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
    --task_name TurnOffMicrowave \
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