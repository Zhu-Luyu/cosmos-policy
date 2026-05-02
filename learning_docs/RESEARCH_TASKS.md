# Cosmos Policy 递进式学习与科研任务清单

这份任务清单的目标是把“学习项目”和“推进科研实验”拆成一系列可以完成的作业。每个任务都有明确产出，做完一个再进入下一个。

建议节奏：

- 每次只做一个任务。
- 每个任务完成后写 5-10 行实验/学习记录。
- 遇到看不懂的地方，不要求完全理解底层实现，先记录问题并继续完成任务。
- 每个阶段结束后再回头整理理解。

## 阶段 0：建立项目地图

### 任务 0.1：画出训练主流程

目标：理解训练入口从命令行到 model forward 的路径。

需要阅读：

- `cosmos_policy/scripts/train.py`
- `cosmos_policy/trainer.py`
- `cosmos_policy/models/policy_text2world_model.py`
- `cosmos_policy/models/policy_video2world_model.py`

要回答的问题：

- 训练命令进入哪个 Python 文件？
- config 在哪里加载？
- model 在哪里 instantiate？
- dataloader 在哪里创建？
- `training_step()` 在哪里被调用？
- loss 最终在哪里返回？

交付物：

- 新建或更新一份笔记：`experiments_notes/00_training_flow.md`
- 用文字或流程图写出：

```text
train.py
-> config
-> model
-> dataloader
-> trainer.train()
-> model.training_step()
-> compute_loss()
```

完成标准：

- 能用自己的话讲清楚一次训练 step 从哪里开始，到哪里算出 loss。

---

### 任务 0.2：整理核心文件职责表

目标：知道以后查问题该去哪找。

交付物：

- 在 `experiments_notes/00_code_map.md` 中写一个表格：

```text
文件路径 | 主要职责 | 我现在理解到什么程度 | 后续要追的问题
```

至少包含：

- `cosmos_policy/scripts/train.py`
- `cosmos_policy/trainer.py`
- `cosmos_policy/models/policy_text2world_model.py`
- `cosmos_policy/models/policy_video2world_model.py`
- `cosmos_policy/experiments/robot/cosmos_utils.py`
- `cosmos_policy/datasets/libero_dataset.py`
- `cosmos_policy/datasets/robocasa_dataset.py`
- `cosmos_policy/datasets/aloha_dataset.py`
- `cosmos_policy/modules/cosmos_sampler.py`
- `cosmos_policy/modules/hybrid_edm_sde.py`

完成标准：

- 看到一个 bug 或实验想法时，大致知道应该先打开哪个文件。

## 阶段 1：跑通最小可观察闭环

### 任务 1.1：跑一次 config dryrun

目标：学会展开配置，知道实际训练参数来自哪里。

命令模板：

```bash
torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_predict2_2b_480p_libero \
  --dryrun
```

如果具体 config 名称不同，以你当前实验实际使用的 config 为准。

要记录：

- dryrun 是否成功。
- 输出的 config.yaml 路径。
- model type 是什么。
- dataset type 是什么。
- batch size 是多少。
- `state_t` 是多少。
- `min_num_conditional_frames` 是多少。
- `action_loss_multiplier` 是多少。
- action masking 相关配置是什么。

交付物：

- `experiments_notes/01_dryrun_config.md`

完成标准：

- 你能从展开后的 config 中找到模型、数据集、训练器、loss mask 相关参数。

---

### 任务 1.2：打印一个 batch 的 key 和 shape

目标：建立对训练数据 batch 的直觉。

做法：

- 在 `CosmosPolicyDiffusionModel.training_step()` 开头临时加入 shape trace。
- 只打印 rank 0。
- 打印一次后可以立刻 `raise SystemExit`，避免真的训练很久。

建议打印：

```python
print("data_batch keys:", data_batch.keys())
for k, v in data_batch.items():
    if hasattr(v, "shape"):
        print(k, tuple(v.shape), v.dtype)
    else:
        print(k, type(v))
```

交付物：

- `experiments_notes/01_batch_shapes.md`
- 粘贴一次真实输出。
- 对每个重要 key 写一句解释。

完成标准：

- 你能指出 batch 中哪些字段对应 image、proprio、action、value、mask、latent index。

## 阶段 2：理解 latent sequence

### 任务 2.1：手动画出 LIBERO 或 RoboCasa 的 latent 布局

目标：理解每一帧 latent 的语义。

需要阅读：

- `MODEL_DESIGN.md`
- `cosmos_policy/experiments/robot/cosmos_utils.py`
- `get_latent_indices_from_model_config()`

交付物：

- `experiments_notes/02_latent_layout.md`

内容模板：

```text
state_t = ?
min_num_conditional_frames = ?

latent frame 0: ?
latent frame 1: ?
latent frame 2: ?
...

condition frames:
denoise frames:
action frame:
value frame:
future state frames:
```

完成标准：

- 不看文档时，你也能写出 action frame 在第几帧、value frame 在第几帧。

---

### 任务 2.2：打印 condition mask

目标：理解哪些帧是条件输入，哪些帧需要去噪。

做法：

- 在 `CosmosPolicyVideo2WorldModel.get_data_and_condition()` 里打印：

```python
mask = condition.condition_video_input_mask_B_C_T_H_W
print("condition mask shape:", mask.shape)
print("condition mask by frame:", mask[0, 0, :, 0, 0])
print("action idx:", data_batch.get("action_latent_idx"))
print("value idx:", data_batch.get("value_latent_idx"))
```

交付物：

- `experiments_notes/02_condition_mask.md`
- 记录 demo sample、world model sample、value function sample 的 mask 差异。

完成标准：

- 你能解释为什么 world model sample 要把 action frame 也设为 condition。
- 你能解释为什么 value function sample 通常只预测 value frame。

## 阶段 3：理解当前 action latent baseline

### 任务 3.1：逐行解释 `replace_latent_with_action_chunk()`

目标：完全理解当前 repeat-fill baseline。

需要阅读：

- `cosmos_policy/models/policy_text2world_model.py`
- `replace_latent_with_action_chunk()`

交付物：

- `experiments_notes/03_action_repeat_fill.md`

必须包含：

- 输入 shape。
- `flat_action` shape。
- `latent_elements` 是多少。
- `num_repeats` 怎么算。
- 最终 action latent frame shape。
- 为什么它叫 repeat-fill。
- 这个方法可能有什么问题。

完成标准：

- 你能不用看代码，自己写出这个函数的伪代码。

---

### 任务 3.2：写一个 action latent toy test

目标：不用跑大模型，也能验证 action 注入逻辑。

做法：

- 新建一个小脚本，例如：

```text
debug/action_latent_toy.py
```

脚本构造：

```python
x0 = torch.zeros(B, 16, T, 28, 28)
action_chunk = torch.arange(...).reshape(B, chunk_size, action_dim)
action_indices = torch.tensor([...])
out = replace_latent_with_action_chunk(x0, action_chunk, action_indices)
```

需要验证：

- action frame 被修改。
- 非 action frame 没有被修改。
- action 值确实按 repeat-fill 填进去。

交付物：

- `debug/action_latent_toy.py`
- `experiments_notes/03_action_toy_test.md`

完成标准：

- 这个脚本能在 CPU 上跑通，不依赖 GPU，不加载大模型。

## 阶段 4：建立 baseline 评估记录

### 任务 4.1：记录原始模型推理流程

目标：理解 inference 如何从 latent 中取回 action。

需要阅读：

- `cosmos_policy/experiments/robot/cosmos_utils.py`
- 搜索 action extraction 相关逻辑。

建议命令：

```bash
rg -n "action|action_chunk|latent_idx|unnormalize" cosmos_policy/experiments/robot/cosmos_utils.py
```

交付物：

- `experiments_notes/04_inference_action_decode.md`

必须回答：

- 推理时 action latent 是在哪个函数生成的？
- denoised latent 里的 action frame 如何变回 action chunk？
- action 是否做了 normalize/unnormalize？
- 如果修改 action encoding，推理侧需要同步改哪里？

完成标准：

- 你能列出训练注入和推理解码之间必须保持一致的地方。

---

### 任务 4.2：跑一次 baseline 小评估或 smoke test

目标：为后续新方法建立对照。

可选目标：

- 如果完整 eval 太贵，先跑一个最小 smoke test。
- 如果已有评估脚本，先跑少量 episode。

交付物：

- `experiments_notes/04_baseline_result.md`

记录：

- checkpoint
- config
- dataset/task
- denoising steps
- 是否启用 future state/value
- action 输出是否合理
- success rate 或最小可观察指标
- 遇到的问题

完成标准：

- 后续每个 action latent 实验都有一个 baseline 可以比较。

## 阶段 5：第一个科研改动：region-fill

### 任务 5.1：设计 region-fill 方案

目标：先写设计，再写代码。

设计问题：

- action latent frame 使用哪个区域？
- 未使用区域填 0、保留原值、还是填 learnable constant？
- action extraction 时从哪里读？
- 是否仍然保持无参数？
- 是否兼容旧 checkpoint？

交付物：

- `experiments_notes/05_region_fill_design.md`

完成标准：

- 设计中明确训练注入和推理解码是互相匹配的。

---

### 任务 5.2：实现 region-fill toy function

目标：先不接入主模型，在 toy script 中验证。

建议新增函数：

```python
replace_latent_with_action_chunk_region_fill(...)
```

先放在 toy script 或单独 debug 文件里，不急着改主模型。

交付物：

- `debug/action_latent_region_fill_toy.py`
- `experiments_notes/05_region_fill_toy.md`

完成标准：

- CPU 上可以验证：
  - action 区域正确填充。
  - 非 action 区域符合设计。
  - 能从该区域恢复原 action chunk。

---

### 任务 5.3：接入主模型并跑 smoke test

目标：把 region-fill 作为第一个真实模型改动。

注意：

- 需要同步修改训练注入和推理解码。
- 建议加 config 开关，例如 `action_latent_encoding = "repeat_fill" | "region_fill"`。
- 默认保持 `"repeat_fill"`，避免破坏 baseline。

交付物：

- 代码改动。
- `experiments_notes/05_region_fill_smoke_test.md`

完成标准：

- baseline config 仍能使用 repeat-fill。
- region-fill config 能跑到至少一个 batch。
- shape、loss、action decode 没有明显错误。

## 阶段 6：第二个科研改动：channel-wise 或 temporal layout

### 任务 6.1：实现 channel-wise action encoding

目标：探索比 region-fill 更结构化的无参数编码。

设计思路：

- 把 action chunk flatten 后按 channel 分配。
- 或者每个 action dim 对应固定 channel/区域。
- 剩余位置 zero padding。

交付物：

- `experiments_notes/06_channelwise_design.md`
- toy test
- smoke test

完成标准：

- 能和 repeat-fill、region-fill 放在同一套 config 开关下切换。

---

### 任务 6.2：实现 temporal layout action encoding

目标：显式保留 action chunk 的时间结构。

设计思路：

- 每个 action step 使用一个 spatial stripe。
- 或每个 action step 使用一组 patch。
- 每个 stripe/patch 内编码该 step 的 action_dim。

交付物：

- `experiments_notes/06_temporal_layout_design.md`
- toy test
- smoke test

完成标准：

- 你能解释这个 layout 如何比 flatten repeat 更保留时间结构。

## 阶段 7：可学习 action encoder/decoder

### 任务 7.1：设计 learnable action encoder/decoder

目标：进入真正结构改造。

设计问题：

- encoder 输入是什么 shape？
- encoder 输出完整 latent frame，还是输出低分辨率 token 再 upsample？
- decoder 从 denoised action latent 如何恢复 action chunk？
- action loss 是 latent MSE，还是 action-space MSE，还是两者都有？
- 新参数如何初始化？
- 旧 checkpoint 如何加载？

交付物：

- `experiments_notes/07_learnable_action_codec_design.md`

完成标准：

- 设计中明确新模块参数、loss、checkpoint 兼容策略。

---

### 任务 7.2：先实现轻量 MLP codec

目标：做最小可学习版本。

建议：

```text
ActionEncoder:
  (B, chunk_size * action_dim)
  -> MLP
  -> (B, C * H * W)
  -> (B, C, H, W)

ActionDecoder:
  (B, C, H, W)
  -> flatten
  -> MLP
  -> (B, chunk_size, action_dim)
```

交付物：

- 代码改动。
- toy overfit test。
- smoke test。
- `experiments_notes/07_mlp_codec_smoke_test.md`

完成标准：

- 不接大模型时，codec 可以 overfit 几个随机 action chunk。
- 接入大模型后能跑一个 batch。

## 阶段 8：实验对比与总结

### 任务 8.1：建立实验对比表

目标：把科研结果变成可比较的证据。

交付物：

- `experiments_notes/action_latent_experiment_table.md`

表格字段：

```text
实验名
encoding 方法
是否有新参数
参数量
训练数据
checkpoint
训练 steps
action MSE
success rate
value 指标
future state 指标
训练是否稳定
备注
```

完成标准：

- 至少包含 baseline、region-fill、一个结构化编码方法。

---

### 任务 8.2：写第一版科研总结

目标：把实验沉淀成论文/报告思路。

交付物：

- `experiments_notes/action_latent_research_summary.md`

内容：

- 原始 repeat-fill 的问题是什么。
- 你提出了哪些 action latent 表示。
- 哪些有效，哪些无效。
- 指标如何变化。
- 可能原因是什么。
- 下一步最值得做什么。

完成标准：

- 这份总结可以直接转化成组会汇报或论文实验小节。

## 每次任务完成后的固定复盘模板

每完成一个任务，在对应笔记末尾写：

```text
## 复盘

我完成了什么：

我现在理解了什么：

我还不理解什么：

遇到的 bug / 坑：

下一步：
```

## 当前推荐从哪里开始

如果现在不知道做什么，就按这个顺序做：

1. 任务 0.1：画出训练主流程。
2. 任务 1.1：跑 config dryrun。
3. 任务 1.2：打印一个 batch 的 key 和 shape。
4. 任务 2.1：画 latent layout。
5. 任务 3.1：逐行解释 repeat-fill。
6. 任务 3.2：写 action latent toy test。

做完这 6 个任务后，再开始第一个科研改动 region-fill。
