# Cosmos Policy + VITA Phase 1 Action Codec 修改说明

本文档用于对齐本次 `feature/action-codec-phase1` 分支的修改主线：为什么改、改了哪里、如何启用、目前验证到哪一步，以及后续在 A20/H20 GPU 服务器上应该怎么继续测试。

## 1. 修改主线

原始 Cosmos Policy 的 action 表示方式比较直接：把 `(chunk_size, action_dim)` 的低维 action chunk flatten，然后重复填满一个视频 latent frame，例如 `(16, 28, 28)`。训练时 DiT 学的是这个手工 repeat-fill latent；推理时再从生成的 action frame 里切出多份 action chunk 并平均。

这次 Phase 1 的目标是保留 Cosmos Policy 的整体范式，但把 action frame 的手工 repeat-fill 替换成可学习的 action latent codec：

```text
raw action chunk A
  -> ActionLatentCodec.encode_frame(A)
  -> learned action latent frame z_a: (B, 16, 28, 28)
  -> 注入 Cosmos latent sequence 的 action frame

generated / denoised action latent z_hat_a
  -> ActionLatentCodec.decode_frame(z_hat_a)
  -> predicted action chunk A_hat
```

这条路线借鉴的是 VITA 的关键思想：不要强行用重复纹理表示 action，而是学习一个 action latent space，让 action latent 与视觉 latent 的形状更匹配。第一阶段不做 VITA 的 unrolled FLD，只加一个轻量的 denoised decode loss，让生成出来的 action latent 能被 decoder 读成真实 action。

## 2. 为什么这样改

主要问题是原 baseline 的 action 维度和 latent frame 维度极不匹配。以 RoboCasa 为例，action chunk 是 `32 * 7 = 224` 维，而 action latent frame 是 `16 * 28 * 28 = 12544` 维。repeat-fill 虽然简单稳定，但会制造大量重复信息和没有空间语义的“action 纹理”。

本次设计保守地只替换 action frame 的编码/解码边界：

- 不改 Cosmos-Predict2 / DiT 主干。
- 不改视频 tokenizer。
- 不改 proprio/value/future image 的表示方式。
- 默认 `use_action_latent_codec=False`，旧 baseline 配置保持原行为。
- 新功能只在新增 RoboCasa action codec 配置里启用。

训练时 encoded action latent 在 diffusion target 里会 `detach()`。这样 EDM latent loss 不会反向推动 encoder 去追随当前模型预测，避免 target latent 自己漂移；codec 的 encoder/decoder 主要通过 action autoencoder loss 学，decoder 还通过 denoised decode loss 接收模型输出侧的监督。

## 3. 关键代码改动

### 新增 codec 模块

文件：`cosmos_policy/models/action_latent_codec.py`

新增 `ActionLatentCodec`：

- 输入 action chunk: `(B, chunk_size, action_dim)`
- 输出 action latent frame: `(B, latent_channels, latent_height, latent_width)`
- 默认设计面向 RoboCasa：`chunk_size=32`、`action_dim=7`、latent frame `(16, 28, 28)`
- 内部结构是 MLP encoder/decoder：
  - encoder: action flatten -> hidden -> bottleneck -> full latent frame
  - decoder: latent frame flatten -> bottleneck -> hidden -> action flatten
- 暴露接口：
  - `encode_frame(action_chunk)`
  - `decode_frame(action_latent_frame)`
  - `forward(action_chunk)`，即 encode 后 decode，用于 AE loss

对应单测文件：`cosmos_policy/models/action_latent_codec_test.py`

### 模型配置扩展

文件：`cosmos_policy/models/policy_text2world_model.py`

`CosmosPolicyModelConfig` 新增字段：

```python
use_action_latent_codec: bool = False
action_codec_chunk_size: Optional[int] = None
action_codec_action_dim: Optional[int] = None
action_codec_latent_height: int = 28
action_codec_latent_width: int = 28
action_codec_bottleneck_dim: int = 512
action_codec_hidden_dim: int = 1024
action_codec_ae_loss_weight: float = 1.0
action_codec_denoised_loss_weight: float = 1.0
```

当 `use_action_latent_codec=True` 时，模型初始化 `self.action_latent_codec`；否则为 `None`，保持 baseline。

### action 注入逻辑

文件：`cosmos_policy/models/policy_text2world_model.py`

`replace_latent_with_action_chunk()` 增加可选参数：

```python
action_latent_codec: Optional[ActionLatentCodec] = None
```

行为：

- `action_latent_codec is None`：保持原来的 repeat-fill。
- `action_latent_codec is not None`：用 `codec.encode_frame(action_chunk).detach()` 生成 action latent frame，再写入 action latent index。

文件：`cosmos_policy/models/policy_video2world_model.py`

`condition.gt_frames` 中的 action conditioning 也改为传入同一个 codec，保证 world model/value function 需要把 action 当条件时，看到的 action frame 和训练 target 的表示一致。

### loss 修改

文件：`cosmos_policy/models/policy_text2world_model.py`

原有 EDM latent loss 保持不变，只是启用 codec 后 action frame 的 target 从 repeat-fill 变成 learned encoded latent。

新增两个 loss：

```text
action_codec_ae_mse_loss
  = MSE(codec.decode_frame(codec.encode_frame(actions)), actions)

action_codec_denoised_mse_loss
  = MSE(codec.decode_frame(model_pred_action_frame), actions)
```

`action_codec_denoised_mse_loss` 只对 demo / policy action samples 计算，也就是 `rollout_data_mask == 0` 的样本。这样它更贴近“policy action prediction”的监督，不把 world model/value sample 混进来。

总训练 loss 在原有 reduced diffusion loss 后加：

```python
action_codec_total_loss =
    action_codec_ae_loss_weight * action_codec_ae_mse_loss
    + action_codec_denoised_loss_weight * action_codec_denoised_mse_loss
```

并写入 `output_batch`：

- `action_codec_ae_mse_loss`
- `action_codec_denoised_mse_loss`
- `action_codec_total_loss`

### W&B logging

文件：`cosmos_policy/config/callbacks.py`

训练和验证阶段都新增两个统计项：

- `action_codec_ae_mse_loss`
- `action_codec_denoised_mse_loss`

W&B key 分别是：

```text
train/action_codec_ae_mse_loss
train/action_codec_denoised_mse_loss
val/action_codec_ae_mse_loss
val/action_codec_denoised_mse_loss
```

如果使用 `wandb_10x` 之类带 tag 的 callback，key 中会带原有的 `@10` 后缀机制。

### 推理解码路径

文件：`cosmos_policy/experiments/robot/cosmos_utils.py`

`extract_action_chunk_from_latent_sequence()` 增加可选参数：

```python
action_latent_codec=None
```

行为：

- `None`：保持原来的 flatten + chunk average。
- 非 `None`：直接取 action latent frame，然后 `codec.decode_frame()` 得到 action chunk。

调用点同步传入：

- `cosmos_policy/experiments/robot/cosmos_utils.py`
- `cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py`
- `cosmos_policy/experiments/robot/aloha/deploy.py`

传入方式统一为：

```python
action_latent_codec=getattr(model, "action_latent_codec", None)
```

这样 baseline 模型没有 codec 时不会受影响。

### RoboCasa 实验配置

文件：`cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`

新增训练配置：

```python
cosmos_predict2_2b_480p_robocasa_50_demos_per_task_action_codec
```

它继承：

```python
cosmos_predict2_2b_480p_robocasa_50_demos_per_task
```

并启用：

```python
use_action_latent_codec=True
action_codec_chunk_size=32
action_codec_action_dim=7
```

新增推理配置：

```python
cosmos_predict2_2b_480p_robocasa_50_demos_per_task_action_codec__inference
```

它继承 action codec 训练配置，并覆盖 inference SDE：

```python
sigma_max=80
sigma_min=4
```

## 4. 分支与远端状态

本地实现分支：

```text
feature/action-codec-phase1
```

已推送到远程：

```text
origin/feature/action-codec-phase1
```

当前实现 commit：

```text
96b6f87 feat: add phase1 action latent codec
```

A20/H20 服务器上的测试目录按要求放在：

```text
/home/CONNECT/yfang870/yunhengwang/zhu_luyu/cosmos-policy-action-codec
```

该目录是从 GitHub 远程分支 clone 下来的，不是 rsync 的本地临时副本，便于保证代码来源统一。

## 5. 已完成的本地验证

本机 Mac 上完成：

```bash
python -m py_compile \
  cosmos_policy/models/action_latent_codec.py \
  cosmos_policy/models/action_latent_codec_test.py \
  cosmos_policy/models/policy_text2world_model.py \
  cosmos_policy/models/policy_video2world_model.py \
  cosmos_policy/experiments/robot/cosmos_utils.py \
  cosmos_policy/experiments/robot/aloha/deploy.py \
  cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py \
  cosmos_policy/config/callbacks.py \
  cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py
```

结果：通过。

```bash
git diff --check
```

结果：通过。

用本机已有 `lerobot` 环境做了 codec smoke：

```text
encode_frame 输出 shape: torch.Size([2, 16, 28, 28])
decode_frame 输出 shape: torch.Size([2, 32, 7])
encoder/decoder 均有梯度
```

本机没有完成项目级 pytest / config dryrun，原因是当前 Mac worktree 的 `uv --no-sync` 环境缺少 `pytest` 和 `loguru`；尝试 `uv run --group dev` 时会开始解析/下载 CUDA、flash-attn 等重依赖，不适合在 Mac 上继续。

## 6. A20/H20 服务器验证计划

服务器别名：

```bash
ssh a20x8-vpn
```

实际 `hostname`：

```text
H20-1
```

`nvidia-smi` 显示是 8 张 H20。按约定，测试只使用两张 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

远端代码目录：

```bash
cd /home/CONNECT/yfang870/yunhengwang/zhu_luyu/cosmos-policy-action-codec
```

建议验证顺序：

### 6.1 单测

```bash
uv run --no-sync pytest cosmos_policy/models/action_latent_codec_test.py -q
```

目标：

- codec encode/decode shape 正确
- codec reconstruction 可反传
- baseline repeat-fill roundtrip 不变
- codec extraction path 可返回 `(B, 32, 7)`

### 6.2 配置 dryrun

```bash
uv run --no-sync python -m cosmos_policy.scripts.train \
  --dryrun \
  --config cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py \
  experiment=cosmos_predict2_2b_480p_robocasa_50_demos_per_task_action_codec
```

目标：

- 新配置能被 LazyConfig 正确解析
- `use_action_latent_codec=True` 生效
- job name 是 `cosmos_predict2_2b_480p_robocasa_50_demos_per_task_action_codec_phase1`

### 6.3 两卡训练 smoke

建议先跑极短训练，例如 2 step，并关闭不必要的 wandb / validation / checkpoint 频率干扰。具体 override 需要根据服务器已有环境和数据路径确认。原则是：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=<free_port> \
  -m cosmos_policy.scripts.train \
  --config cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py \
  experiment=cosmos_predict2_2b_480p_robocasa_50_demos_per_task_action_codec \
  trainer.max_iter=2 \
  trainer.run_validation=False
```

需要确认：

- 训练能实例化 model / dataloader / optimizer
- loss 是 finite
- output batch 中出现：
  - `action_codec_ae_mse_loss`
  - `action_codec_denoised_mse_loss`
- 只占用两张卡，不启动 8 卡任务

## 7. 当前未完成事项

截至本文档写入时，A20/H20 服务器上已经完成：

- 创建个人实验目录：
  - `/home/CONNECT/yfang870/yunhengwang/zhu_luyu/`
- 从远程 GitHub 分支 clone 代码：
  - `/home/CONNECT/yfang870/yunhengwang/zhu_luyu/cosmos-policy-action-codec`
- 远端 clone 的 HEAD 对齐到：
  - `96b6f87`

尚未完成：

- 远端 pytest
- 远端 config dryrun
- 远端两卡 training smoke

原因是正在探测服务器上可用的 Python/uv/conda 环境；非交互 SSH shell 默认 PATH 里没有 `python`、`conda`、`uv`，已经定位到：

```text
/home/CONNECT/yfang870/.local/bin/uv
/home/CONNECT/yfang870/miniconda3/bin/conda
/home/CONNECT/yfang870/miniconda3/bin/python
```

后续应该用绝对路径或显式 source conda 初始化脚本继续跑，避免依赖交互 shell。

## 8. 风险与后续优化

本次 Phase 1 是最小闭环，不是最终论文级版本。需要关注：

- action codec 目标 latent 当前是随机初始化后 joint learning，前期 decoder 重构可能较差。
- action latent target 使用 `detach()`，更稳定，但 encoder 只通过 AE loss 学，不直接被 diffusion loss 推动。
- `action_codec_denoised_mse_loss` 会给 decoder 和 DiT 输出侧共同施压，若权重过大可能影响 diffusion latent loss，需要看训练曲线。
- 当前没有独立 action AE 预训练流程。后续可以加 A1 实验：先预训练/freeze codec，再训练 Cosmos。
- 当前没有 unrolled FLD。若 Phase 1 有收益，再考虑 Phase 2 的 sampler unroll。

建议第一组实验对比：

```text
A0: 原 repeat-fill baseline
A2: joint action codec + diffusion loss + AE loss
A3: A2 + denoised action decode loss（当前分支）
```

优先观察：

- normalized action MSE
- action L1 / per-dim error
- `action_codec_ae_mse_loss`
- `action_codec_denoised_mse_loss`
- inference latency
- RoboCasa success rate 小样本趋势
