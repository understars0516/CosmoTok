# -*- coding: utf-8 -*-
import os
import sys

from lightning.pytorch.cli import LightningCLI

if __name__ == "__main__":
    # 通过命令行:
    #   python train.py --model model_name_or_path --data data_name_or_path ...
    # 例如:
    #   python train.py --model rg.model.demo.LitAutoEncoder --data rg.model.demo.LitMnistDataModule ...
    # 实现任意模型与数据组合训练.
    # 另外, 可以通过命令行查看模型和数据可传入的参数, 例如
    #   python train.py --data.help rg.model.demo.LitMnistDataModule
    #   python train.py --model.help rg.model.demo.LitAutoEncoder
    # 最后, optimizers, schedulers 等对象均可通过类似的方式在命令行进行设置.
    # see https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html

    # 1. 当使用 torchrun 启动该脚本时, TORCHELASTIC_RUN_ID 始终会被设置 (参见
    #    https://pytorch.org/docs/stable/elastic/run.html#environment-variables). 此时, 我们始终应该为命令行添加 fit 参数,
    #    使得 torchrun 启动的每个进程都被设置了该参数.
    # 2. 直接使用 python train.py 启动该脚本时, 只需要为启动进程设置 fit 参数即可. 启动脚本启动的其他进程会自动继承该参数,
    #    如果也添加该参数, 会导致子命令重复.
    # 为了简单, 加以直接使用 python 执行该启动脚本, 而不再使用 torchrun 启动.
    if os.environ.get("LOCAL_RANK", None) is None or os.environ.get("TORCHELASTIC_RUN_ID", None) is not None:
        sys.argv.insert(1, "fit")
    LightningCLI(save_config_callback=None)
