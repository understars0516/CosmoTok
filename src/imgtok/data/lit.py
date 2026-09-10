import typing

import lightning as ltn
from imgtok.data.datasets import SPDLAstroImageDatasetAION, SPDLAstroImageDatasetStats, SPDLCosmogridDataset
from torch.utils.data import DataLoader

if typing.TYPE_CHECKING:
    from jsonargparse.typing import Path_drw


# see https://lightning.ai/docs/pytorch/stable/data/datamodule.html for how to create a custom datamodule
class ImageDataModuleAION(ltn.LightningDataModule):
    # noinspection PyTypeHints
    def __init__(
            self,
            root_dir: "Path_drw",
            batch_size: int = 128,
            num_threads: int = 64,
            num_cpu_workers: int = 8,
            num_io_workers: int = 16,
            num_dataloader_workers: int = 0,
    ):
        """HSC Image data

        :param root_dir: root directory to store the HSC Image data.
        :param batch_size: batch size.
        """
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_split = None
        self.valid_split = None
        self.test_split = None
        self.dataloader_common_kwargs = {
            "collate_fn": None,
            "pin_memory": True,
            "batch_size": None,
            "num_workers": num_dataloader_workers,
            # "persistent_workers": True,
            # "multiprocessing_context": "spawn",
        }
        if num_dataloader_workers > 0:
            self.dataloader_common_kwargs["persistent_workers"] = True
            self.dataloader_common_kwargs["multiprocessing_context"] = "spawn"

        self.dataset_common_kwargs = {
            "batch_size": self.batch_size,
            "num_threads": num_threads,
            "num_cpu_workers": num_cpu_workers,
            "num_io_workers": num_io_workers,
            "num_dataloader_workers": num_dataloader_workers,
        }

    def prepare_data(self):
        # Downloading and saving data with multiple processes (distributed settings) will result in corrupted data.
        # Lightning ensures the prepare_data() is called only within a single process on CPU, so you can safely add
        # your downloading logic within. In case of multi-node training, the execution of this hook depends upon
        # prepare_data_per_node. setup() is called after prepare_data and there is a barrier in between which ensures
        # that all the processes proceed to setup once the data is prepared and available for use.
        # 因为数据已经下载好，所以 prepare_data() 函数中什么也不做
        pass

    def setup(self, stage: str) -> None:
        # 注意, 这个函数中禁用了 MNIST 数据集的下载功能 (download=False) 因为在 DDP 模式下, 该函数会被每个进程调用.
        # 多个进程下载同一个数据, 明显存在问题. 因此下载数据, 预处理等工作是交给 prepare_data() 函数的.
        if stage == "fit":
            self.train_split = SPDLAstroImageDatasetAION(self.root_dir, "train",
                                                         **self.dataset_common_kwargs, shuffle=True)
            self.valid_split = SPDLAstroImageDatasetAION(self.root_dir, "valid", **self.dataset_common_kwargs)
            print(f"train dataset size: {len(self.train_split)}, valid dataset size: {len(self.valid_split)}")
        else:
            self.test_split = SPDLAstroImageDatasetAION(self.root_dir, "test", **self.dataset_common_kwargs)

    def train_dataloader(self) -> DataLoader:
        # 使用 Lightning Trainer, dataloader 的 sampler 和 shuffle 都不用再设置了.
        # 因为 Trainer 会自动添加合适的 sampler 和 shuffle.
        # see:
        #   https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#remove-samplers
        #   https://lightning.ai/docs/pytorch/stable/common/trainer.html#lightning.pytorch.trainer.Trainer.params.use_distributed_sampler
        return DataLoader(self.train_split, shuffle=False, **self.dataloader_common_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_split, **self.dataloader_common_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_split, **self.dataloader_common_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_split, **self.dataloader_common_kwargs)

    # @staticmethod
    # def collate_fn(batch: "list[ImageDatum]") -> ImageDatum:
    #     instances = [(x.flux, x.masked_ivar) for x in batch]
    #     flux, masked_ivar = torch.utils.data.default_collate(instances)
    #     return ImageDatum(flux, masked_ivar)


# see https://lightning.ai/docs/pytorch/stable/data/datamodule.html for how to create a custom datamodule
class ImageDataModuleStats(ltn.LightningDataModule):
    # noinspection PyTypeHints
    def __init__(
            self,
            root_dir: "Path_drw",
            batch_size: int = 128,
            num_threads: int = 64,
            num_cpu_workers: int = 8,
            num_io_workers: int = 16,
            num_dataloader_workers: int = 0,
            disable_ivar: bool = False,
    ):
        """HSC Image data

        :param root_dir: root directory to store the HSC Image data.
        :param batch_size: batch size.
        """
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_split = None
        self.valid_split = None
        self.test_split = None
        self.dataloader_common_kwargs = {
            "collate_fn": None,
            "pin_memory": True,
            "batch_size": None,
            "num_workers": num_dataloader_workers,
            # "persistent_workers": True,
            # "multiprocessing_context": "spawn",
        }
        if num_dataloader_workers > 0:
            self.dataloader_common_kwargs["persistent_workers"] = True
            self.dataloader_common_kwargs["multiprocessing_context"] = "spawn"

        self.dataset_common_kwargs = {
            "batch_size": self.batch_size,
            "num_threads": num_threads,
            "num_cpu_workers": num_cpu_workers,
            "num_io_workers": num_io_workers,
            "num_dataloader_workers": num_dataloader_workers,
            "disable_ivar": disable_ivar,
        }

    def prepare_data(self):
        # Downloading and saving data with multiple processes (distributed settings) will result in corrupted data.
        # Lightning ensures the prepare_data() is called only within a single process on CPU, so you can safely add
        # your downloading logic within. In case of multi-node training, the execution of this hook depends upon
        # prepare_data_per_node. setup() is called after prepare_data and there is a barrier in between which ensures
        # that all the processes proceed to setup once the data is prepared and available for use.
        # 因为数据已经下载好，所以 prepare_data() 函数中什么也不做
        pass

    def setup(self, stage: str) -> None:
        # 多个进程下载同一个数据, 明显存在问题. 因此下载数据, 预处理等工作是交给 prepare_data() 函数的.
        if stage == "fit":
            self.train_split = SPDLAstroImageDatasetStats(self.root_dir, "train",
                                                          **self.dataset_common_kwargs, shuffle=True)
            self.valid_split = SPDLAstroImageDatasetStats(self.root_dir, "valid", **self.dataset_common_kwargs)
            print(f"train dataset size: {len(self.train_split)}, valid dataset size: {len(self.valid_split)}")
        else:
            self.test_split = SPDLAstroImageDatasetStats(self.root_dir, "test", **self.dataset_common_kwargs)

    def train_dataloader(self) -> DataLoader:
        # 使用 Lightning Trainer, dataloader 的 sampler 和 shuffle 都不用再设置了.
        # 因为 Trainer 会自动添加合适的 sampler 和 shuffle.
        # see:
        #   https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#remove-samplers
        #   https://lightning.ai/docs/pytorch/stable/common/trainer.html#lightning.pytorch.trainer.Trainer.params.use_distributed_sampler
        return DataLoader(self.train_split, shuffle=False, **self.dataloader_common_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_split, **self.dataloader_common_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_split, **self.dataloader_common_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_split, **self.dataloader_common_kwargs)

    # @staticmethod
    # def collate_fn(batch: "list[ImageDatum]") -> ImageDatum:
    #     instances = [(x.flux, x.masked_ivar) for x in batch]
    #     flux, masked_ivar = torch.utils.data.default_collate(instances)
    #     return ImageDatum(flux, masked_ivar)


class CosmoGridDataModule(ltn.LightningDataModule):
    # noinspection PyTypeHints
    def __init__(
            self,
            root_dir: "Path_drw",
            batch_size: int = 128,
            num_threads: int = 32,
            num_cpu_workers: int = 8,
            num_io_workers: int = 4,
            num_dataloader_workers: int = 0,
    ):
        """HSC Image data

        :param root_dir: root directory to store the HSC Image data.
        :param batch_size: batch size.
        """
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_split = None
        self.valid_split = None
        self.test_split = None
        self.dataloader_common_kwargs = {
            "collate_fn": None,
            "pin_memory": True,
            "batch_size": None,
            "num_workers": num_dataloader_workers,
        }
        self.dataset_common_kwargs = {
            "batch_size": self.batch_size,
            "num_threads": num_threads,
            "num_cpu_workers": num_cpu_workers,
            "num_io_workers": num_io_workers,
            "num_dataloader_workers": num_dataloader_workers,
        }

    def prepare_data(self):
        # Downloading and saving data with multiple processes (distributed settings) will result in corrupted data.
        # Lightning ensures the prepare_data() is called only within a single process on CPU, so you can safely add
        # your downloading logic within. In case of multi-node training, the execution of this hook depends upon
        # prepare_data_per_node. setup() is called after prepare_data and there is a barrier in between which ensures
        # that all the processes proceed to setup once the data is prepared and available for use.
        # 因为数据已经下载好，所以 prepare_data() 函数中什么也不做
        pass

    def setup(self, stage: str) -> None:
        # 注意, 这个函数中禁用了 MNIST 数据集的下载功能 (download=False) 因为在 DDP 模式下, 该函数会被每个进程调用.
        # 多个进程下载同一个数据, 明显存在问题. 因此下载数据, 预处理等工作是交给 prepare_data() 函数的.
        if stage == "fit":
            self.train_split = SPDLCosmogridDataset(self.root_dir, "train",
                                                    **self.dataset_common_kwargs, shuffle=True)
            self.valid_split = SPDLCosmogridDataset(self.root_dir, "valid", **self.dataset_common_kwargs)
            print(f"train dataset size: {len(self.train_split)}, valid dataset size: {len(self.valid_split)}")
        else:
            self.test_split = SPDLCosmogridDataset(self.root_dir, "test", **self.dataset_common_kwargs)

    def train_dataloader(self) -> DataLoader:
        # 使用 Lightning Trainer, dataloader 的 sampler 和 shuffle 都不用再设置了.
        # 因为 Trainer 会自动添加合适的 sampler 和 shuffle.
        # see:
        #   https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#remove-samplers
        #   https://lightning.ai/docs/pytorch/stable/common/trainer.html#lightning.pytorch.trainer.Trainer.params.use_distributed_sampler
        return DataLoader(self.train_split, **self.dataloader_common_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_split, **self.dataloader_common_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_split, **self.dataloader_common_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_split, **self.dataloader_common_kwargs)

    # @staticmethod
    # def collate_fn(batch: "list[ImageDatum]") -> ImageDatum:
    #     instances = [(x.flux, x.masked_ivar) for x in batch]
    #     flux, masked_ivar = torch.utils.data.default_collate(instances)
    #     return ImageDatum(flux, masked_ivar)
