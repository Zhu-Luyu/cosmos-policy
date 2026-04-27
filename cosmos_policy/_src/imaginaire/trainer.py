# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import os
import signal

import torch
import torch.distributed as dist
import torch.utils.data

from cosmos_policy._src.imaginaire.flags import INTERNAL
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
from cosmos_policy._src.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
    print("Megatron-core is not installed.")


from cosmos_policy._src.imaginaire.lazy_config import LazyConfig, instantiate
from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.utils import callback, distributed, ema, log, misc
from cosmos_policy._src.imaginaire.utils.checkpointer import Checkpointer
from cosmos_policy._src.imaginaire.utils.misc import StragglerDetectorV2


class ImaginaireTrainer:
    """The base trainer class of Imaginaire.

    All trainers in Imaginaire should inherit ImaginaireTrainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    中文导读：
    这个类是 Imaginaire/Cosmos Policy 的“通用训练控制器”。它不关心具体模型如何算 loss，
    而是负责把一次训练任务所需的基础设施串起来：
    - 分布式初始化：让多 GPU / 多进程之间能通信。
    - Megatron parallel_state：记录 tensor/pipeline/context/data parallel 的分组关系。
    - 日志和配置保存：保证实验可复现。
    - callback：在训练关键节点插入日志、监控、保存可视化等扩展逻辑。
    - checkpoint：加载/保存模型、优化器、scheduler、GradScaler 状态。
    - 训练循环：dataloader -> forward -> backward -> optimizer step -> validation/checkpoint。

    如果你熟悉 ResNet 单卡训练，可以把它理解为把下面这些常见代码统一封装了：
    model.cuda(); optimizer = ...; for batch in loader: loss.backward(); optimizer.step()
    只是这里额外处理了分布式训练、混合精度、梯度累积、EMA、profiling 和 checkpoint。

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (misc.Timer): Timer object to time code blocks and functions.
    """

    def __init__(self, config):
        """Constructor of the trainer.

        Args:
            config (Config): The config object for the Imaginaire codebase.
        """
        super().__init__()
        self.config = config
        # Set up the distributed computing environment.
        # 初始化分布式通信环境。torchrun 启动时会给每个进程分配 rank/local_rank/world_size 等环境变量，
        # distributed.init() 会读取这些信息并调用 torch.distributed.init_process_group。
        # 即使外层 train.py 已经初始化过，这里再调用一次通常会由项目封装处理成幂等操作。
        with distributed_init():
            distributed.init()
            # Set up parallel states.
            # 兼容旧配置：早期把 context_parallel_size 放在 config.model 里，
            # 现在推荐统一放在 config.model_parallel.context_parallel_size。
            if hasattr(config.model, "context_parallel_size"):
                if config.model_parallel.context_parallel_size > 1:
                    raise ValueError(
                        "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                        "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                    )
                else:
                    log.critical(
                        "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                    )
                    config.model_parallel.context_parallel_size = config.model.context_parallel_size
            if USE_MEGATRON:
                # Megatron 的 parallel_state 会把所有 rank 划分成不同并行组。
                # 常见组包括：
                # - tensor model parallel：把单层里的大矩阵/attention 计算切到多张 GPU。
                # - pipeline model parallel：把不同网络层放到不同 GPU。
                # - context parallel：把长序列/上下文维度切到多张 GPU。
                # - data parallel：多个完整/分片模型副本吃不同数据，再同步梯度。
                #
                # 后续 train.py 里的 DistributedSampler 会用 data_parallel_rank/world_size，
                # 因为“全局 rank”不一定等于“数据并行 rank”。
                if (
                    "create_gloo_process_groups"
                    in inspect.signature(parallel_state.initialize_model_parallel).parameters
                ):
                    # 不同版本 megatron-core 的 initialize_model_parallel 参数略有差异，
                    # 这里用 inspect 做版本兼容。
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                        create_gloo_process_groups=False,
                    )
                else:
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                    )
                # `config.model_parallel.sequence_parallel` is a bool that indicates whether to use sequence parallelism.
                # It is not part of the original `parallel_state` API, so we need to set it manually.
                # sequence_parallel 是项目额外挂到 parallel_state 上的标志。
                # 它通常配合 tensor parallel 使用，用来进一步切分 Transformer 中和 sequence 维度相关的中间激活。
                parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
                if parallel_state.sequence_parallel:
                    # Megatron 的序列并行通常要求设置这个 CUDA 环境变量，
                    # 目的是限制 CUDA 连接数，避免某些通信/计算重叠模式下出现性能或正确性问题。
                    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Create the local job directory, save the config file, and pipe to a local log.
        # rank0 是主进程。多进程训练时，保存配置、创建目录这类操作通常只让 rank0 做，
        # 否则所有 rank 同时写同一个文件，容易产生竞争或重复日志。
        if distributed.is_rank0():
            os.makedirs(config.job.path_local, exist_ok=True)
            # Save the config as .pkl for reproducibility.
            LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
            # Save the config as .yaml for reading or parsing experiment hyperparameters.
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        # barrier 表示所有 rank 在这里集合等待。确保 rank0 创建目录/保存配置完成后，
        # 其他 rank 再继续执行后面的日志初始化或 checkpoint 逻辑。
        dist.barrier()
        if INTERNAL:
            log.init_loguru_file(f"{config.job.path_local}/stdout.log")
            if distributed.is_rank0():
                # Print important environment variables and the effective config.
                log.info("Config:\n" + config.pretty_print(use_color=True))
            misc.print_environ_variables(["TORCH_HOME", "IMAGINAIRE_OUTPUT_ROOT", "ENABLE_ONELOGGER"])
        else:
            misc.print_environ_variables(["HF_HOME", "IMAGINAIRE_OUTPUT_ROOT"])
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        # by_rank=True 表示不同 rank 会得到不同随机种子。
        # 这对数据增强、dropout、随机采样等很重要，避免多个 rank 做出完全相同的随机选择。
        misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
        # Initialize cuDNN.
        # cuDNN deterministic=True 更强调可复现性；benchmark=True 会为固定输入尺寸搜索更快算法。
        # 两者的取舍由配置决定。
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # Floating-point precision settings.
        # 允许 TF32。Ampere 及更新 GPU 上，TF32 可以明显加速 matmul/conv，
        # 数值精度低于 FP32 但通常足够训练深度模型。
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        # Initialize the callback functions.
        # callback 是训练流程的“钩子系统”。例如：
        # on_train_start、on_before_forward、on_after_backward、on_validation_end。
        # 具体 callback 可以做日志、W&B 上报、可视化、梯度裁剪、EMA 更新等。
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # Initialize the model checkpointer.
        # checkpointer 负责恢复和保存训练状态。保存的不只是模型权重，
        # 还包括 optimizer、scheduler、grad_scaler 和当前 iteration，这样可以断点续训。
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            # 如果配置里指定了自定义 checkpoint 类，则通过 LazyConfig instantiate 创建。
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        # Initialize the timer for speed benchmarking.
        # training_timer 用来给 dataloader/forward/backward/optimizer_step 等阶段计时，
        # 方便判断瓶颈在数据读取、模型计算还是优化器更新。
        self.training_timer = misc.TrainingTimer()
        # Initialize Straggler Detection
        # Straggler 指“慢 rank”。分布式训练里所有 rank 经常需要同步，
        # 如果某一张 GPU/某个 dataloader worker 很慢，其他 rank 都会等它。
        # 这个 detector 用来定位 dataloading、forward、backward、optimizer 哪一段拖慢了整体。
        self.straggler_detector = StragglerDetectorV2(
            enabled=self.config.trainer.straggler_detection.enabled,
            report_freq=self.config.trainer.straggler_detection.report_freq,
            profile_freq=self.config.trainer.straggler_detection.profile_freq,
            max_diff=self.config.trainer.straggler_detection.max_diff,
            raise_error=self.config.trainer.straggler_detection.raise_error,
        )
        self.straggler_detector.initialize()
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        # 训练步超时保护。如果某一步卡住超过 timeout_period，会触发 timeout_handler，
        # 比无限挂住更容易暴露死锁、dataloader 卡死或通信卡死问题。
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))  # type: ignore

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        # 把模型搬到 GPU，并按配置设置 memory_format。
        # memory_format 可能是 contiguous_format 或 channels_last；后者常用于提高卷积模型性能。
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        # 给具体模型一个训练开始前的初始化机会，例如初始化 EMA、缓存、编译模块、设置 tokenizer 等。
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        # callback 包住 optimizer 初始化，方便记录耗时或在初始化前后插入额外逻辑。
        self.callbacks.on_optimizer_init_start()
        # 具体模型决定如何根据 config.optimizer/config.scheduler 创建优化器和学习率调度器。
        # 这让 trainer 不需要知道模型内部哪些参数需要 weight decay、哪些参数要冻结等细节。
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        # GradScaler 用于混合精度训练。它会把 loss 放大后 backward，降低 fp16 梯度 underflow 风险；
        # 如果检测到 inf/nan，也会跳过 optimizer step 并自动调整 scale。
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        # 加载 checkpoint。如果是从头训练，通常返回 iteration=0；
        # 如果断点续训，会恢复模型/优化器/scheduler/scaler 状态，并返回保存时的 iteration。
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        # grad_accum_iter 记录当前处在第几个梯度累积 micro-batch。
        # 例如 grad_accum_iter 配置为 4 时，连续 4 个 batch backward 后才 optimizer.step 一次。
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            # DDP(Data Distributed Parallel) 会在 backward 时自动跨 data parallel ranks 同步梯度。
            # 注意 model_ddp 是 wrapper，原始模型通常在 model_ddp.module 里。
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            # FSDP(Fully Sharded Data Parallel) 会把参数、梯度、优化器状态切分到多个 GPU，
            # 更省显存。这里假设模型已经在别处按 FSDP 方式准备好，所以直接使用 model。
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting training...")
        # 通知所有 callback 训练正式开始。
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        # 有些实验会在 iteration=0 先跑一次验证，作为 fine-tune 前的 baseline。
        if self.config.trainer.run_validation and iteration == 0 and self.config.trainer.run_validation_on_start:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        # profiling/memory snapshot 是可选诊断工具。
        # torch_profiler 可以记录 CUDA kernel、CPU op、通信等耗时；
        # memory_profiler 可以记录显存分配快照，排查 OOM 或显存泄漏。
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            # 外层 while 表示 epoch 级循环。这里没有显式 epoch 计数，
            # 每次 dataloader 走到 StopIteration 后重新创建 iterator，直到达到 max_iter。
            while True:
                dataloader_train_iter = iter(dataloader_train)
                # 内层 while 消耗当前 dataloader iterator 中的 batch。
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        # 读取一个 batch，并给 dataloader 阶段计时/做慢 rank 分析。
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        # 当前 dataloader 被读完，跳出内层循环，外层会重新创建 iterator。
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    # If max_iter is reached, exit the training loop.
                    # iteration 只在真正执行 optimizer.step 后递增，因此 max_iter 指“优化器更新次数”，
                    # 不是 dataloader 取 batch 的次数。开启梯度累积时，两者不同。
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    # misc.to 会递归处理 dict/list/tuple，把其中的 tensor 搬到 CUDA。
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    # 训练步前后的 callback 通常用于日志、可视化、学习率记录、EMA 更新等。
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    # grad_accum_iter != 0 表示还没有凑够配置要求的 micro-batch 数，
                    # 此时已经 backward 了，但还没有 optimizer.step，也不递增 iteration。
                    if grad_accum_iter != 0:
                        continue
                    # Do the following when an actual optimizer (update) step has been made.
                    # 只有真正执行了一次 optimizer.step，才算完成一个 training iteration。
                    iteration += 1
                    # Save checkpoint.
                    # 按固定 iteration 间隔保存 checkpoint。
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    # 按固定 iteration 间隔跑验证。
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # This iteration is successful; reset the timeout signal.
                    # 当前 iteration 成功结束后重置超时闹钟。下一步如果卡住过久，会触发前面注册的 signal handler。
                    signal.alarm(self.config.trainer.timeout_period)
                    self.straggler_detector.generate_report(iteration)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                if _end_training:
                    break
        log.success("Done with training.")
        # 如果最后一个 iteration 刚好没有落在 save_iter 上，训练结束前再保存一次最终状态。
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        # finalize 可能会上传/整理 checkpoint 元数据，或关闭 checkpoint 后端资源。
        self.checkpointer.finalize()
        # 训练结束前同步所有 rank，确保没有某些进程提前退出导致其他进程通信报错。
        distributed.barrier()
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        # 梯度累积时，前几个 micro-batch 只在本 rank 本地累积梯度，不做 DDP all-reduce；
        # 到累积窗口的最后一个 micro-batch，才同步梯度。这样可以减少通信次数。
        #
        # 例：grad_accum_iter=4
        # micro-batch 1/2/3: backward 但不同步梯度
        # micro-batch 4: backward 并同步梯度，然后 optimizer.step()
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            # forward 阶段调用具体模型的 training_step。
            # 对 Cosmos Policy 来说，loss 的具体计算逻辑在模型类里，不在 trainer 里。
            with self.training_timer("forward"):
                with self.straggler_detector.profile_section(
                    "fwd", self.config.trainer.straggler_detection.analyze_forward
                ):
                    output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            # backward 阶段：loss 缩放、反向传播、模型级 backward 后处理。
            with self.training_timer("backward"):
                with self.straggler_detector.profile_section(
                    "bwd", self.config.trainer.straggler_detection.analyze_backward
                ):
                    # loss 除以 grad_accum_iter，是为了让累积 N 个 micro-batch 后的梯度平均值
                    # 和一次性用大 batch 训练时的梯度尺度保持一致。
                    loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                    loss_scaled.backward()
                    # 给模型一个 backward 后处理机会，例如梯度检查、梯度裁剪前准备、统计信息等。
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_after_backward()
                    else:
                        model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            # 累积够指定 micro-batch 数后，才真正更新一次参数。
            with self.training_timer("optimizer_step"):
                with self.straggler_detector.profile_section(
                    "opt", self.config.trainer.straggler_detection.analyze_optimizer
                ):
                    self.callbacks.on_before_optimizer_step(
                        model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                    )
                    # GradScaler 会先 unscale 梯度，检查 inf/nan，再决定是否调用 optimizer.step。
                    grad_scaler.step(optimizer)
                    # 根据本次是否溢出，更新下一步使用的 scale。
                    grad_scaler.update()
                    # 更新学习率调度器。这里是每次 optimizer update 后 step 一次。
                    scheduler.step()
                    self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                    # 给模型一个清梯度前的 hook，常用于 EMA 更新、日志统计或自定义状态维护。
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                    else:
                        model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                    # set_to_none=True 比把梯度清零更省显存/更快；下一次 backward 会重新创建 grad tensor。
                    optimizer.zero_grad(set_to_none=True)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

    @torch.no_grad()
    def validate(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """Validate on the full validation dataset.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        # 切换到 eval 模式，影响 dropout、batchnorm 等模块行为。
        model.eval()
        # Evaluate on the full validation set.
        # EMA(Exponential Moving Average) 是模型参数的滑动平均版本。
        # 验证时临时切换到 EMA 权重，通常会让指标更稳定；退出 context 后恢复训练权重。
        with ema.ema_scope(model, enabled=model.config.ema.enabled):
            for val_iter, data_batch in enumerate(dataloader_val):
                # max_val_iter 用于只跑一部分验证集，降低大规模训练时的验证开销。
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                data_batch = misc.to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                # 具体验证逻辑仍然交给模型实现，trainer 只负责循环和 hook。
                output_batch, loss = model.validation_step(data_batch, iteration)
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        self.callbacks.on_validation_end(model, iteration=iteration)
