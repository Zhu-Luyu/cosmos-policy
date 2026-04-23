eval "$(conda shell.bash hook)"
conda activate cosmos-env
cd ~/cosmos-policy

# 1. 确保环境变量指向国内镜像和你的本地模型目录
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HOME/models"
export TRANSFORMERS_CACHE="$HOME/models"

# 1.5 硬要调系统的 ldconfig，就写一个伪造的 ldconfig 脚本拦截它，骗过 transformer_engine，并把它指向我们 .venv 里已经下好的动态库！
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export PATH="$HOME/cosmos-policy/bin:$PATH"

# 2. 运行评估脚本 (注意：我加上了 --no-sync，并修改了本地权重路径)
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