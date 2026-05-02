# Cosmos Policy 学习与科研实验计划

本文档用于记录我后续学习 Cosmos Policy 项目、理解模型各模块输入输出、以及围绕 action latent 表示进行科研实验的路线。

## 1. 当前目标

我的目标不是单纯跑通项目，而是通过“干中学”的方式理解代码和模型：

- 理解模型整体流程：数据集、condition、latent 组装、扩散训练、采样推理。
- 理解各模块输入输出：尤其是 action、proprio、image、value 如何统一进入 latent video sequence。
- 通过修改模型结构做科研实验，重点研究 action latent 的编码方式是否会影响性能。
- 当前最重要的实验方向：替换现有 action latent 的重复平铺模式，尝试更合理的 action-to-latent 表示。

## 2. 项目阅读顺序

这个项目不适合一开始从 `_src/` 目录逐行读起。`_src/` 主要是 Cosmos Predict2、Imaginaire、Reason1 等底座代码，调用栈深且抽象多。建议先把它当作黑盒。

优先阅读这些文件：

1. `CODEGUIDE.md`
   - 项目结构、核心思想、训练流程的导读。

2. `MODEL_DESIGN.md`
   - latent sequence 设计、WAN VAE、action/proprio/value 注入方式、condition mask、DiT 输入输出。

3. `cosmos_policy/scripts/train.py`
   - 训练入口。
   - 重点看 config 加载、model 实例化、dataset/dataloader 创建、trainer 调用。

4. `cosmos_policy/models/policy_text2world_model.py`
   - Cosmos Policy 的核心模型逻辑。
   - 重点看：
     - `replace_latent_with_action_chunk()`
     - `replace_latent_with_proprio()`
     - `CosmosPolicyDiffusionModel.training_step()`
     - `compute_loss()`
     - sampling 相关函数。

5. `cosmos_policy/models/policy_video2world_model.py`
   - Video2World 条件注入和 condition mask 逻辑。
   - 重点看：
     - `CosmosPolicyVideo2WorldModel.get_data_and_condition()`
     - 哪些 latent frame 被当作条件输入
     - world model / value function 样本如何改变 mask。

6. `cosmos_policy/experiments/robot/cosmos_utils.py`
   - 推理侧核心逻辑。
   - 重点看：
     - `get_latent_indices_from_model_config()`
     - `get_action()`
     - action / future state / value 如何从 latent 中解码出来。

7. `cosmos_policy/datasets/`
   - 数据如何组织成训练 batch。
   - 重点看 batch 里有哪些 key，以及每个 key 的 shape。

## 3. 模型主线理解

Cosmos Policy 的核心设计是把机器人控制问题统一成视频扩散问题。

原始输入包括：

- 当前图像观测
- 当前 proprioception
- 语言任务描述
- 训练时的 ground-truth action chunk
- 训练时的 future state
- 训练时的 value/return

这些信息会被组装成统一的 latent video sequence：

```text
LIBERO:
[blank, curr_proprio, curr_wrist_img, curr_primary_img,
 action, future_proprio, future_wrist_img, future_primary_img, value]

RoboCasa / ALOHA:
[blank, curr_proprio, curr_wrist_img, curr_img2, curr_primary_img,
 action, future_proprio, future_wrist_img, future_img2, future_primary_img, value]
```

其中：

- image frame 通过 WAN 2.1 VAE 编码成 latent。
- proprio/action/value 本身不是图像，但会被填充进同样形状的 latent frame。
- 条件帧是已知观测，采样时会被固定。
- action、future state、value 是模型需要预测或去噪的帧。

典型 latent shape：

```text
(B, C, T, H, W)
C = 16
H = W = 28
T = 9  或 11
```

## 4. HPC 上的调试策略

目前 HPC 上不能方便使用 VSCode 单步调试，因此建议使用 tmux + debug 节点 + pdb/日志的方式。

申请 debug 节点：

```bash
tmux new -s cosmos-debug
srun -p debug --gres=gpu:1 --pty bash
cd /path/to/cosmos-policy
```

先使用 dryrun 展开配置：

```bash
torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_predict2_2b_480p_libero \
  --dryrun
```

如果要进入交互式调试，可以在关键函数中临时加入：

```python
breakpoint()
```

推荐插入位置：

- `cosmos_policy/models/policy_text2world_model.py`
  - `training_step()`
  - `compute_loss()`
  - `replace_latent_with_action_chunk()`

- `cosmos_policy/models/policy_video2world_model.py`
  - `get_data_and_condition()`

- `cosmos_policy/experiments/robot/cosmos_utils.py`
  - `get_action()`

进入 pdb 后优先看这些内容：

```python
data_batch.keys()
{k: v.shape for k, v in data_batch.items() if hasattr(v, "shape")}
latent_state.shape
condition.condition_video_input_mask_B_C_T_H_W.shape
data_batch["action_latent_idx"]
data_batch["value_latent_idx"]
```

调试重点不是逐行走完整个大模型，而是抓住每个阶段的：

- 输入是什么
- 输出是什么
- shape 是什么
- mask 如何变化
- action/proprio/value 被放到了哪个 latent frame
- loss 具体作用在哪些帧上

## 5. 推荐的“干中学”路线

### 阶段 1：只观察，不改行为

目标：建立对数据流和 shape 的直觉。

建议添加临时 shape trace，打印：

- `data_batch` 的所有 key 和 shape
- `raw_state` / `latent_state` shape
- `condition_video_input_mask_B_C_T_H_W`
- action/proprio/value 的 latent index
- loss 分项

这一阶段不要急着改模型。先确认自己能说清楚：

- 一个 batch 进入模型时包含什么
- action latent 在第几帧
- 条件帧和去噪帧分别有哪些
- loss 是在哪些 latent frame 上计算的

### 阶段 2：改 loss 和 mask

目标：理解多任务训练机制。

优先研究：

- `mask_loss_for_action_future_state_prediction`
- `mask_value_prediction_loss_for_policy_prediction`
- `mask_current_state_action_for_value_prediction`
- `mask_future_state_for_qvalue_prediction`
- `action_loss_multiplier`

这类实验风险较低，适合作为第一个科研改动：

- 提高 action loss 权重
- 只训练 action prediction
- action + future state 联合训练
- value loss 是否作为辅助项

### 阶段 3：改 action latent 表示

目标：围绕当前科研目标，替换 action 的重复平铺模式。

当前代码位置：

```text
cosmos_policy/models/policy_text2world_model.py
replace_latent_with_action_chunk()
```

当前做法：

```text
action_chunk: (B, chunk_size, action_dim)
flatten:      (B, chunk_size * action_dim)
repeat fill:  (B, 16 * 28 * 28)
reshape:      (B, 16, 28, 28)
```

这个方法简单直接，但存在潜在问题：

- action 信息只是机械重复，没有空间结构。
- latent frame 中大量位置包含重复值，信息冗余很强。
- 图像 latent 的空间归纳偏置可能不适合这种平铺信号。
- DiT 看到的是类似“伪图像”的 action frame，但这个 frame 并不具备真实图像的局部空间语义。

因此可以设计新的 action latent representation。

## 6. Action Latent 科研实验方向

### Baseline：现有 repeat-fill

这是必须保留的 baseline。

```text
action vector -> flatten -> repeat -> fill entire latent frame
```

所有新方法都应该和它比较：

- success rate
- rollout performance
- action MSE
- value prediction quality
- training stability
- convergence speed

### 方案 A：区域填充

只把 action 填入 latent frame 的一部分区域，其余区域置零或保留 learnable mask。

例子：

```text
latent frame: (16, 28, 28)
只使用左上角区域或若干 channel 存储 action
```

可能优点：

- 减少重复冗余。
- 让模型更容易区分 action token 区域和空白区域。

需要注意：

- 推理时提取 action 的逻辑也要对应修改。
- 如果 action 区域太小，可能表达能力不足。

### 方案 B：channel-wise action encoding

把不同 action 维度映射到不同 channel，而不是在所有 channel 和空间位置上重复平铺。

例子：

```text
channel 0-6: 当前 action dim
channel 7-13: 后续 action dim 或 chunk summary
其他 channel: padding / learnable / zero
```

可能优点：

- 更符合 latent channel 的表示方式。
- action 维度之间更容易分离。

问题：

- action chunk 有时间维度，chunk_size * action_dim 可能超过 16 个 channel，需要设计时间维度压缩方式。

### 方案 C：learnable MLP projection

用一个小 MLP 把 action chunk 投影成完整 latent frame。

```text
action_chunk: (B, chunk_size, action_dim)
flatten:      (B, chunk_size * action_dim)
MLP:          (B, 16 * 28 * 28)
reshape:      (B, 16, 28, 28)
```

可能优点：

- 表达能力强。
- 能学习 action 到 latent 空间的更自然映射。

问题：

- 参数量可能较大。
- 需要同步设计 inverse/extraction 方式。
- 如果直接从 denoised latent 反推出 action，需要一个 decoder head 或固定反投影方式。

适合的变体：

```text
action encoder: action -> latent frame
action decoder: latent frame -> action
```

这会把 action latent 从“手工填充”变成一个可学习 bottleneck。

### 方案 D：action token + broadcast

先把 action chunk 编码成少量 action tokens，再通过固定或可学习方式 broadcast 到 latent frame。

```text
action_chunk -> MLP/Transformer -> K action tokens
action tokens -> reshape/upsample -> latent frame
```

可能优点：

- 比直接 MLP 到完整 latent 更轻。
- 能保留 action chunk 的时间结构。

问题：

- 实现复杂度比 repeat-fill 高。
- 需要设计 token 到 latent frame 的布局。

### 方案 E：时间结构编码

当前 repeat-fill 会把整个 action chunk flatten，时间结构不明显。可以显式保留 chunk 时间顺序。

可能做法：

- 每个 action step 占据 latent frame 的一个空间 stripe。
- 每个 action step 占据一组 spatial patch。
- 使用 1D temporal conv / transformer 编码 action chunk，再映射到 latent frame。

可能优点：

- 对 action chunk prediction 更自然。
- 适合研究 chunk 内时间依赖。

问题：

- 需要仔细设计 action extraction。

## 7. 推荐实验顺序

不要一开始就做最复杂的 learnable projection。建议按风险从低到高推进。

1. Baseline 复现
   - 跑通原始 repeat-fill。
   - 保存训练日志、评估结果、action MSE、success rate。

2. 区域填充
   - 最小改动。
   - 主要验证“减少重复冗余”是否有帮助。

3. channel-wise encoding
   - 仍然是规则编码，不引入太多可学习参数。
   - 适合和区域填充对比。

4. temporal layout encoding
   - 保留 action chunk 时间结构。
   - 适合研究动作序列建模。

5. learnable action encoder/decoder
   - 表达能力最强，但变量也最多。
   - 适合作为后续重点方法。

## 8. 实验实现时必须同步修改的位置

修改 action latent 表示时，不能只改训练时的注入函数。至少要检查这些位置：

1. 训练时 action 注入

```text
cosmos_policy/models/policy_text2world_model.py
replace_latent_with_action_chunk()
```

2. 推理时 action 生成与提取

```text
cosmos_policy/experiments/robot/cosmos_utils.py
```

需要找到从 denoised action latent frame 中恢复 action chunk 的逻辑。

3. loss 计算

```text
cosmos_policy/models/policy_text2world_model.py
compute_loss()
```

需要确认 action loss 是在 latent frame 上算，还是解码成 action 后算。如果只在 latent frame 上算，那么新编码方式会直接改变 loss 空间。

4. dataset stats / normalize / unnormalize

```text
cosmos_policy/datasets/
cosmos_policy/experiments/robot/cosmos_utils.py
```

需要确认 action 在进入 latent 前是否 normalized，以及推理输出是否 unnormalized。

5. checkpoint 兼容性

如果引入 learnable encoder/decoder：

- 旧 checkpoint 可能缺少新参数。
- 需要处理 strict loading 或初始化新模块。
- 可能需要记录新 config 字段，避免实验不可复现。

## 9. 每次实验应记录的信息

每个 action latent 实验都应该记录：

- 实验名称
- git commit 或 patch
- config 名称
- 数据集和任务
- action encoding 方法
- 是否引入新参数
- 新参数量
- action latent shape
- action extraction 方法
- training loss 曲线
- action MSE
- success rate
- value prediction 指标
- future state prediction 质量
- 是否出现训练不稳定或 NaN

建议每个实验单独写一个短日志，例如：

```text
experiments_notes/action_latent_region_fill.md
experiments_notes/action_latent_channelwise.md
experiments_notes/action_latent_mlp.md
```

## 10. 最小可行任务

下一步最有价值的任务不是直接大改模型，而是先写一个调试脚本或 trace 工具：

目标：

- 跑一个 batch。
- 打印 `data_batch` 所有 key 和 shape。
- 打印 action/proprio/value latent index。
- 打印 action 注入前后的 latent 统计量。
- 打印 condition mask。
- 可选：保存 action latent frame 的可视化或数值摘要。

这个工具可以帮助我在每次修改 action encoding 后快速确认：

- 编码是否正确
- shape 是否正确
- 推理提取是否和训练注入一致
- 新方法是否真的改变了 action latent 的结构

## 11. 总结

后续学习和实验的核心方法：

```text
读一个函数
-> 打印输入输出和 shape
-> 跑一个 batch
-> 画出/记录 latent 布局
-> 做一个小改动
-> 验证 loss、action 输出和评估指标
```

科研主线：

```text
现有 repeat-fill action latent
-> 规则化 action latent layout
-> 保留 action 时间结构
-> learnable action encoder/decoder
-> 比较性能和训练稳定性
```

当前优先级最高的具体方向：

1. 建立 action latent trace 工具。
2. 复现 baseline。
3. 实现 region-fill 或 channel-wise action encoding 作为第一个对比实验。
4. 确认训练注入和推理解码完全一致。
