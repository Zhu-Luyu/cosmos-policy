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

"""
Cosmos Policy training script with manual DistributedSampler instantiation.

This script extends the base training script to manually create DistributedSampler
instead of using instantiate(), avoiding duplicate dataset creation.

整体流程：
1. 从命令行读取 Python LazyConfig 配置文件和覆盖项。
2. 初始化分布式环境，让 Megatron / torch.distributed 知道当前进程的 rank。
3. 根据 config 创建 trainer 和 model。
4. 手动创建带 DistributedSampler 的训练/验证 DataLoader。
5. 把 model 和 dataloader 交给 trainer.train()，真正的训练循环在 trainer 内部执行。
"""

import argparse
import os
import traceback

from loguru import logger as logging
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.config import Config, load_config, pretty_print_overrides
from cosmos_policy._src.imaginaire.lazy_config import LazyConfig, instantiate
from cosmos_policy._src.imaginaire.serialization import to_yaml
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_policy._src.imaginaire.utils.launch import log_reproducible_setup


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    """启动一次完整训练任务。

    这里做的是“训练前的组装工作”：分布式初始化、配置校验、创建 trainer/model/dataloader。
    真正的 forward、loss、backward、optimizer.step、checkpoint、validation 等循环逻辑，
    在 config.trainer.type(config) 返回的 trainer 对象里实现。
    """
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    # 先初始化分布式环境。config.validate() 里面可能会跨 rank 同步 tensor / buffer，
    # 所以必须先让 torch.distributed 和 Megatron 的 parallel_state 准备好。
    # 典型启动方式是 torchrun，每个 GPU 对应一个 Python 进程。
    with distributed_init():
        distributed.init()

    # Check that the config is valid
    # 校验配置是否合法，例如字段类型、必填项、并行参数等。
    config.validate()
    # Freeze the config so developers don't change it during training.
    # 冻结配置，避免训练过程中某段代码意外修改超参，导致实验不可复现。
    config.freeze()  # type: ignore

    # config.trainer.type 是 LazyConfig 里配置的 trainer 类，通常是 CosmosPolicyTrainer。
    # 这里实例化 trainer 后，会在 trainer.__init__ 中继续初始化 Megatron model parallel、
    # 保存 config、设置随机种子、构建 callback、checkpoint manager 等。
    trainer = config.trainer.type(config)

    # Setup the miscellaneous stuff for reproducibility.
    # 记录可复现实验所需的信息，例如命令行参数、git 状态、环境等。
    log_reproducible_setup(config, args)

    # 按 config.model 的 LazyCall 实例化模型。
    # model_init() 本身主要包了一层计时/telemetry，不改变模型构造逻辑。
    with model_init():
        model = instantiate(config.model)

    # Create the dataloaders.
    with data_loader_init():
        # NOTE (user): We manually instantiate the dataloader instead of using instantiate(config.dataloader_train),
        # since it is difficult to set up the DistributedSampler without creating two duplicates of the dataset.
        # We intentionally instantiate the dataloader on every process (rather than the rank 0 process only) to work with the DistributedSampler.
        # 这里没有直接 instantiate(config.dataloader_train)，而是只 instantiate dataset，
        # 再手动塞进 DistributedSampler 和 DataLoader。
        #
        # 原因：分布式训练时，每个 rank 都需要同一个 dataset 的不同切片。
        # 如果让 LazyConfig 直接构造完整 dataloader，再额外替换 sampler，很容易重复创建 dataset，
        # 对大数据集、远程数据集或有状态数据集都不划算，也可能引入副作用。
        dataset = instantiate(config.dataloader_train.dataset)

        # DistributedSampler 负责把 dataset 按 data parallel rank 切分。
        # 例如 data parallel world size = 8 时，每个 rank 只遍历约 1/8 数据。
        # 这里用 Megatron 的 data_parallel rank/world_size，而不是全局 rank/world_size，
        # 因为 tensor/context/pipeline parallel 的多个进程可能共享同一份 data parallel 身份。
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            seed=0,
        )

        # 真正的 PyTorch DataLoader。batch_size 等参数仍然来自配置文件，
        # 只有 sampler 是这个脚本手动接管的。
        # 注意：这里的 batch_size 是“每个 data parallel rank 的 local batch size”，
        # 全局有效 batch size 还要乘以 data parallel size 和 gradient accumulation steps。
        dataloader_train = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=config.dataloader_train.drop_last,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
            pin_memory_device=config.dataloader_train.pin_memory_device,
            timeout=config.dataloader_train.timeout,
        )

        dataloader_val = None
        if config.trainer.run_validation:
            # NOTE (user): Manually instantiate the val dataloader as well
            # 如果配置要求训练中做 validation，也用同样方式手动构造验证 dataloader。
            # 验证集不 shuffle，保证每次验证的样本顺序稳定，便于对比指标和排查问题。
            dataset_val = instantiate(config.dataloader_val.dataset)
            sampler_val = DistributedSampler(
                dataset=dataset_val,
                num_replicas=parallel_state.get_data_parallel_world_size(),
                rank=parallel_state.get_data_parallel_rank(),
                shuffle=False,  # Do not shuffle the validation set
                seed=0,
            )
            dataloader_val = DataLoader(
                dataset=dataset_val,
                sampler=sampler_val,
                batch_size=config.dataloader_val.batch_size,
                drop_last=config.dataloader_val.drop_last,
                num_workers=config.dataloader_val.num_workers,
                persistent_workers=config.dataloader_val.persistent_workers,
                pin_memory=config.dataloader_val.pin_memory,
                pin_memory_device=config.dataloader_val.pin_memory_device,
                timeout=config.dataloader_val.timeout,
            )

    # Start training
    # 进入 trainer 的训练主循环。
    # 之后的逻辑包括：model.to(cuda)、optimizer/scheduler/grad scaler 初始化、
    # checkpoint 加载、DDP/FSDP 包装、dataloading、training_step、validation、保存模型等。
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train --config=cosmos_policy/config/experiment/your_config.py

    # Get the config file from the input arguments.
    # 这个脚本通常通过 torchrun 启动，这样每个 GPU 会对应一个独立进程：
    # torchrun --nproc_per_node=8 -m cosmos_policy.scripts.train --config=...
    #
    # 也可以单进程调试：
    # torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train --config=...
    parser = argparse.ArgumentParser(description="Training")

    # Python LazyConfig 配置文件路径。该配置文件会返回一个 Config 对象，
    # 里面包含 job、trainer、model、optimizer、scheduler、dataloader 等所有训练组件。
    parser.add_argument("--config", help="Path to the config file", required=False)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    # dryrun 只加载、打印、保存 config，不进入 launch()。
    # 用途：检查命令行 override 是否生效，或确认 LazyConfig 展开后的最终配置。
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()

    # load_config 会读取 --config 指向的 Python 配置，并应用 opts 里的覆盖项。
    # enable_one_logger=True 表示启用统一日志设置，方便多 rank 日志管理。
    config = load_config(args.config, args.opts, enable_one_logger=True)

    if args.dryrun:
        # dryrun 模式下只输出配置并保存到 job.path_local/config.yaml。
        # job.path_local 默认受 IMAGINAIRE_OUTPUT_ROOT 控制，否则落到 /tmp/imaginaire4-output。
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            # 优先使用项目自定义 serialization.to_yaml，它通常能更好地处理 attrs/LazyConfig 对象。
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            # 如果自定义序列化失败，退回 LazyConfig 自带的 YAML 保存逻辑，至少保证 dryrun 有输出文件。
            logging.error("to_yaml failed, falling back to LazyConfig.save_yaml:")
            logging.error(f"Traceback: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
