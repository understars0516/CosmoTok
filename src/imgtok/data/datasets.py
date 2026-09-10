import json
import os
import typing
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import torch
from astro_utils.helper import center_crop, get_ddp_info, get_ddp_worker_info
from imgtok.common import ImageDatasetAttrs, ImageDatum
from imgtok.model.prepare import AstroImagePreprocessorAdHoc, AstroImagePreprocessorStats
from intervaltree import IntervalTree
from spdl.pipeline import PipelineBuilder
from torch.utils.data import IterableDataset

if typing.TYPE_CHECKING:
    from typing import Generator, Iterable, Literal

    from intervaltree import Interval
    from spdl.pipeline import Pipeline


class SpdlIterableDataset(IterableDataset, ABC):
    """
    将 SPDL 的数据 Pipeline 封装为 PyTorch 的 `IterableDataset` 类型的方法. 该类的对象保存了构建数据 pipeline 需要的参数.
    """
    def __init__(
            self,
            # 数据集存储位置, 子集类别
            data_root: str | os.PathLike,
            split: 'Literal["train", "valid", "test"]',
            # 数据集每个元素大小. 与 PyTorch 的 dataset 不同, 使用 SPDL 时, 通常在 dataset 中进行 batch 化.
            batch_size: int = 128,
            # 数据预处理需要的相关资源量, DDP 训练时, 这些值是针对每个数据加载器进程而言的.
            num_threads: int = 64,
            num_cpu_workers: int = 2,
            num_io_workers: int = 4,
            # 构建数据 pipeline 需要的一些通用参数
            sink_buffer_size: int = 4,
            num_dataloader_workers: int = 0,
            shuffle_buffer_batches: int = 4,
            shuffle: bool = False,
            seed: int = 42,
    ) -> None:
        """初始化 SpdlIterableDataset 类的对象

        Args:
            data_root: 数据所在目录
            split: 数据集子集类别, 可选值为: train, valid, test
            batch_size: batch 大小. 该数据集的每个元素都是数据样本的一个 batch, 与 PyTorch 的 dataset 不同, 使用 SPDL 时,
              通常在 dataset 中进行 batch 化.
            num_threads: 执行 SPDL 数据 pipeline 时, 使用的线程数, 即启动的异步事件循环的线程数.
            num_cpu_workers: CPU bound 的操作使用的线程数.
            num_io_workers: IO bound 的操作使用的线程数.
            sink_buffer_size: SPDL Pipeline 的 sink buffer 大小
            num_dataloader_workers: PyTorch 训练时, 将该 dataset 传给 Dataloader, 设置的 num_workers 参数.
              即创建的 Dataloader 时, 用于加载数据的 worker 数量. 在这里仅用于估计数据集大小.
              因为使用多个子进程加载数据时, 每个 worker 仅分配到数据集的一部分. 不提供该参数, 我们将无法计算据集大小.
              要求用户自己保证传给 Dataloader 的 num_workers 参数与该参数相等. 通常使用 SPDL 时, 设置为 0.
              即在训练的主进程中创建多个线程执行数据加载过程, 这样做开销小, 并且并不慢.
            shuffle_buffer_batches: 进行 shuffle 时, 使用流式获取到的 shuffle_buffer_batches * batch_size 个样本执行打乱.
            shuffle: 是否对数据进行打乱
            seed: 随机数种子.
        """
        super().__init__()
        num_dataloader_workers = num_dataloader_workers or 1

        self.data_root = Path(data_root)
        self.split = split
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        self.sink_buffer_size = sink_buffer_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_batches * self.batch_size
        self.base_seed = seed
        self.num_dataloader_workers = num_dataloader_workers

        # 获取 DDP 分布式信息
        self.world_size, self.rank, _ = get_ddp_info()

        self.size: int = 0
        self.sources: list[range] = []

    def __len__(self):
        return self.size

    def __iter__(self):
        # 因为数据中有损坏的需要忽略, 估计的数据集大小可能偏大. 这会导致该迭代器返回的数据量小于期望,
        # 在训练模型时, 一个 epoch 获取不到期望的 batch 数量, 导致进度条不能正确更新.
        # 更严重的是, 不同训练进程产生的数据量不同时, 会导致死锁.
        # 为了避免这些问题, 我们重复地构建 pipeline, 使得生成的数据量始终能够达到期望的数量.
        size = 0

        while size < self.size:
            shard = self.create_job_id_shard()

            # 构建并启动 SPDL Pipeline
            pipeline = self.build_pipeline(shard)

            # 启动后台线程
            # 使用 auto_stop() 确保迭代结束或异常时能自动清理
            with pipeline.auto_stop():
                for item in pipeline.get_iterator():
                    yield item
                    size += 1
                    if size >= self.size:
                        break

    @abstractmethod
    def build_pipeline(self, job_ids):
        pass

    @staticmethod
    def shuffle_examples(jobs: "list[ImageDatum]") -> "Generator[int]":
        rng = np.random.default_rng()
        rng.shuffle(jobs)
        yield from jobs

    # @staticmethod
    # @abstractmethod
    # def batchify(batch: list):
    #     pass

    def create_job_id_shard(self):
        # 根据 DDP 信息，对数据源进行切片
        all_sources = []
        # 获取 DataLoader worker 信息 (如果 num_workers > 0)
        total_workers, worker_id = get_ddp_worker_info()

        if self.shuffle:
            seed = self.base_seed + self.rank * 1000
            if worker_id is not None:
                seed += worker_id * 100
            rng = np.random.default_rng(seed)
            self.base_seed += 10000
            rng.shuffle(self.sources)

        for src in self.sources:
            # 组合分片逻辑：先按 rank 分，再按 worker 分
            shard = src if total_workers is None else src[worker_id::total_workers]
            all_sources.append(shard)
        return chain.from_iterable(all_sources)


class JsonIndexedDataset(SpdlIterableDataset):
    def __init__(
            self,
            # 数据集存储位置, 子集类别
            data_root: str | os.PathLike,
            split: 'Literal["train", "valid", "test"]',
            # 数据集每个元素大小. 与 PyTorch 的 dataset 不同, 使用 SPDL 时, 通常在 dataset 中进行 batch 化.
            batch_size: int = 128,
            # 数据预处理需要的相关资源量, DDP 训练时, 这些值是针对每个数据加载器进程而言的.
            num_threads: int = 64,
            num_cpu_workers: int = 2,
            num_io_workers: int = 4,
            # 构建数据 pipeline 需要的一些通用参数
            sink_buffer_size: int = 4,
            num_dataloader_workers: int = 0,
            shuffle_buffer_batches: int = 4,
            shuffle: bool = False,
            seed: int = 42,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            batch_size=batch_size,
            num_threads=num_threads,
            num_cpu_workers=num_cpu_workers,
            num_io_workers=num_io_workers,
            sink_buffer_size=sink_buffer_size,
            num_dataloader_workers=num_dataloader_workers,
            shuffle_buffer_batches=shuffle_buffer_batches,
            shuffle=shuffle,
            seed=seed,
        )

        index_file = Path(data_root) / f"{split}-index.json"
        with open(index_file, "rt", encoding="utf-8") as fp:
            index_array = json.load(fp)
        self.index = IntervalTree.from_tuples(index_array)

        self.size = self.estimate_dataset_size(index_array)
        self.sources = self.collect_job_ids(index_array)

    @abstractmethod
    def estimate_dataset_size(self, index_array):
        raise NotImplementedError

    def collect_job_ids(self, index_array):
        world_size: int = self.world_size
        rank = self.rank

        res = []
        for start, end, _ in index_array:
            job_ids = range(start, end)[rank::world_size]  # 根据 rank 进行数据分片
            res.append(job_ids)
        return res


class SPDLAstroImageDatasetBase(JsonIndexedDataset):
    def __init__(
            self,
            # 数据集存储位置, 子集类别
            data_root: str | os.PathLike,
            split: 'Literal["train", "valid", "test"]',
            # 数据集每个元素大小. 与 PyTorch 的 dataset 不同, 使用 SPDL 时, 通常在 dataset 中进行 batch 化.
            batch_size: int = 128,
            # 数据预处理需要的相关资源量, DDP 训练时, 这些值是针对每个数据加载器进程而言的.
            num_threads: int = 64,
            num_cpu_workers: int = 2,
            num_io_workers: int = 4,
            # 构建数据 pipeline 需要的一些通用参数
            disable_ivar: bool = False,
            crop_size: int = 96,
            sink_buffer_size: int = 4,
            num_dataloader_workers: int = 0,
            shuffle_buffer_batches: int = 4,
            shuffle: bool = False,
            seed: int = 42,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            batch_size=batch_size,
            num_threads=num_threads,
            num_cpu_workers=num_cpu_workers,
            num_io_workers=num_io_workers,
            sink_buffer_size=sink_buffer_size,
            num_dataloader_workers=num_dataloader_workers,
            shuffle_buffer_batches=shuffle_buffer_batches,
            shuffle=shuffle,
            seed=seed,
        )

        self.crop_size = crop_size
        self.disable_ivar = disable_ivar

    @abstractmethod
    def build_pipeline(self, job_ids) -> "Pipeline":
        raise NotImplementedError

    def estimate_dataset_size(self, index_array):
        world_size: int = self.world_size
        batch_size: int = self.batch_size
        num_workers: int = self.num_dataloader_workers

        size = np.zeros(world_size, dtype=np.int32)
        for rank in range(world_size):
            for start, end, (_, _, chunk_size) in index_array:
                job_ids = range(start, end)[rank::world_size]  # 根据 rank 进行数据分片
                size[rank] += len(job_ids) * chunk_size
        return np.min(size) // batch_size // num_workers

    def collect_job_ids(self, index_array):
        world_size: int = self.world_size
        rank = self.rank

        res = []
        for start, end, _ in index_array:
            job_ids = range(start, end)[rank::world_size]  # 根据 rank 进行数据分片
            res.append(job_ids)
        return res

    def read_chunk(self, i: int) -> ImageDatum | None:
        attr_cls = ImageDatasetAttrs
        res = self.index[i]
        if len(res) == 0:
            return None
        interval: "Interval" = next(iter(res))
        start_idx_this_file = interval.begin
        end_idx_this_file = interval.end
        filepath = self.data_root / "content" / interval.data[0]

        num_chunks = interval.data[1]
        chunk_size = interval.data[2]
        assert end_idx_this_file - start_idx_this_file == num_chunks

        chunk_start_idx = i - start_idx_this_file
        assert 0 <= chunk_start_idx < end_idx_this_file
        with h5py.File(filepath, "r") as h5_group:
            bands = [x.upper().decode() for x in h5_group[attr_cls.KEY_BAND][0]]

            flux = h5_group[attr_cls.KEY_FLUX]
            ivar = h5_group[attr_cls.KEY_IVAR]
            mask = h5_group[attr_cls.KEY_MASK]
            self.validate_shape_chunks(flux, ivar, mask, chunk_size)

            instance_idx = chunk_start_idx * chunk_size
            attrs = ImageDatasetAttrs.get_concrete_attrs(bands)
            indicator = h5_group[attrs.KEY_FILTER][instance_idx:instance_idx + chunk_size]
            valid_indices = attrs.validate(indicator)
            if (num_valid := len(valid_indices)) == 0:
                return None

            # flux = flux[instance_idx:instance_idx + chunk_size]
            # ivar = ivar[instance_idx:instance_idx + chunk_size]
            # mask = mask[instance_idx:instance_idx + chunk_size]
            flux = center_crop(flux, self.crop_size, slice(instance_idx, instance_idx + chunk_size))
            mask = center_crop(mask, self.crop_size, slice(instance_idx, instance_idx + chunk_size))

            if self.disable_ivar:
                ivar = None
            else:
                ivar = center_crop(ivar, self.crop_size, slice(instance_idx, instance_idx + chunk_size))

            if num_valid == flux.shape[0]:
                return ImageDatum(flux, ivar, mask, bands=bands)

            if ivar is not None:
                ivar = ivar[valid_indices]
            return ImageDatum(flux[valid_indices], ivar, mask[valid_indices], bands=bands)

    @staticmethod
    def unbatch(x: ImageDatum) -> "Generator[ImageDatum]":
        batch_size = x.batch_size
        if batch_size > 0:
            for i in range(batch_size):
                yield x[i]
        else:
            yield x

    @staticmethod
    def batchify(batch: list[ImageDatum]) -> ImageDatum:
        if batch[0].ivar is None:
            instances = [(x.flux, x.mask, x.channel_mask) for x in batch]
            flux, mask, channel_mask = torch.utils.data.default_collate(instances)
            ivar = None
        else:
            instances = [(x.flux, x.ivar, x.mask, x.channel_mask) for x in batch]
            flux, ivar, mask, channel_mask = torch.utils.data.default_collate(instances)
        return ImageDatum(flux, ivar, mask, channel_mask)

    @staticmethod
    def validate_shape_chunks(flux, ivar, mask, chunk_size: int):
        flux_shape = flux.shape
        ivar_shape = ivar.shape
        mask_shape = mask.shape

        # 判断维数是否正确
        if not (len(flux_shape) == len(ivar_shape) == 4):
            raise ValueError(f"flux, ivar must have 4 dimensions. "
                             f"Got flux.shape = {flux_shape}, ivar.shape = {ivar_shape}")
        # 判断形状是否正确
        if flux_shape != ivar_shape:
            raise ValueError(f"flux and ivar must have the same shape. "
                             f"Got flux.shape = {flux_shape}, ivar.shape = {ivar_shape}")

        # 对 mask 的形状根据维数进行判断: 要求空间维度相同，batch 维度相同。
        if not (mask_shape[0] == flux_shape[0] and mask_shape[-2:] == flux_shape[-2:]):
            raise ValueError(f"mask and flux must have the same shape on batch and spatial dimensions. "
                             f"Got mask.shape = {mask_shape}, flux.shape = {flux_shape}")

        # 要求在batch维度 chunk 尺寸相同
        if not (chunk_size % flux.chunks[0] == 0 and chunk_size % ivar.chunks[0] == 0 and
                chunk_size % mask.chunks[0] == 0):
            raise ValueError(
                f"flux, ivar, and mask must have incompatible chunk size with chunk size {chunk_size}. "
                f"Got flux.chunks = {flux.chunks}, ivar.chunks = {ivar.chunks}, mask.chunks = {mask.chunks}")


class SPDLAstroImageDatasetAION(SPDLAstroImageDatasetBase):
    def __init__(
            self,
            data_root: str | os.PathLike,
            split: 'Literal["train", "valid", "test"]',
            batch_size: int = 128,
            range_compression_factor: float = 0.01,
            mult_factor: float = 10.0,
            crop_size: int = 96,
            num_threads: int = 64,
            num_cpu_workers: int = 2,
            num_io_workers: int = 4,
            sink_buffer_size: int = 4,
            num_dataloader_workers: int = 0,
            shuffle_buffer_batches: int = 4,
            shuffle: bool = False,
            seed: int = 42,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            batch_size=batch_size,
            num_threads=num_threads,
            num_cpu_workers=num_cpu_workers,
            num_io_workers=num_io_workers,
            sink_buffer_size=sink_buffer_size,
            num_dataloader_workers=num_dataloader_workers,
            shuffle_buffer_batches=shuffle_buffer_batches,
            shuffle=shuffle,
            seed=seed,
        )

        self.preprocessor = AstroImagePreprocessorAdHoc(range_compression_factor, mult_factor, crop_size)

    def build_pipeline(self, job_ids):
        builder = (
            PipelineBuilder()
            .add_source(job_ids)
            .pipe(self.read_chunk, concurrency=self.num_io_workers, name="read_chunk")
            .pipe(self.preprocessor.standardize, concurrency=self.num_cpu_workers, name="standardize")
            .pipe(self.preprocessor.normalize_ivar, concurrency=self.num_cpu_workers, name="normalize_ivar")
            .pipe(self.preprocessor.clamp, concurrency=self.num_cpu_workers, name="clamp")
            .pipe(self.preprocessor.rescale, concurrency=self.num_cpu_workers, name="rescale")
            .pipe(self.preprocessor.range_compress, concurrency=self.num_cpu_workers, name="range_compress")
            .pipe(self.preprocessor.pad_image_channels, concurrency=self.num_cpu_workers, name="pad_image_channels")
            .pipe(self.unbatch, concurrency=1, name="unbatch")
        )
        if self.shuffle:
            builder = (
                builder
                .aggregate(self.shuffle_buffer_size, drop_last=False)
                .pipe(self.shuffle_examples, concurrency=1, name="shuffle_examples")
            )
        return (
            builder
            .aggregate(self.batch_size, drop_last=True)
            .pipe(self.batchify, concurrency=2, name="batchify")
            .add_sink(self.sink_buffer_size)
            .build(num_threads=self.num_threads)
        )

    def profile_pipeline(self):
        from spdl.pipeline import profile_pipeline
        from spdl.pipeline.defs import Aggregate, Pipe, PipelineConfig, SinkConfig, SourceConfig

        job_ids = self.create_job_id_shard()
        pipeline_config = PipelineConfig(
            src=SourceConfig(job_ids),
            pipes=[
                # Aggregate(self.shuffle_buffer_size, drop_last=False),
                # Pipe(self.shuffle_examples, concurrency=6, name="shuffle_examples"),
                Pipe(self.read_chunk, concurrency=self.num_io_workers, name="read_chunk"),
                Pipe(self.preprocessor.standardize, concurrency=1, name="standardize"),
                Pipe(self.preprocessor.normalize_ivar, concurrency=1, name="normalize_ivar"),
                Pipe(self.preprocessor.clamp, concurrency=2, name="clamp"),
                Pipe(self.preprocessor.rescale, concurrency=2, name="rescale"),
                Pipe(self.preprocessor.range_compress, concurrency=2, name="range_compress"),
                Pipe(self.preprocessor.pad_image_channels, concurrency=2, name="pad_image_channels"),
                Pipe(self.unbatch, concurrency=2, name="unbatch"),
                Aggregate(self.batch_size, drop_last=True),
                Pipe(self.batchify, concurrency=4, name="batchify"),
            ],
            sink=SinkConfig(buffer_size=self.sink_buffer_size),
        )

        results = profile_pipeline(pipeline_config, num_inputs=self.batch_size * 10)

        # The results list contains ProfileResult objects for each pipe stage
        for result in results:
            print(f"Stage: {result.name}")
            for stat in result.stats:
                print(f"  Concurrency {stat.concurrency}: "
                      f"QPS={stat.qps:.2f}, "
                      f"Occupancy={stat.occupancy_rate:.2f}")


class SPDLCosmogridDataset(JsonIndexedDataset):
    def __init__(
            self,
            data_root: Path,
            split: 'Literal["train", "valid", "test"]',
            log_mean: float = -4.9179,
            log_stddev: float = 0.4947,
            batch_size: int = 128,
            num_threads: int = 32,
            num_cpu_workers: int = 8,
            num_io_workers: int = 4,
            sink_buffer_size: int = 4,
            num_dataloader_workers: int = 0,
            shuffle_buffer_batches: int = 10,
            shuffle: bool = False,
            seed: int = 42,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            batch_size=batch_size,
            num_threads=num_threads,
            num_cpu_workers=num_cpu_workers,
            num_io_workers=num_io_workers,
            sink_buffer_size=sink_buffer_size,
            num_dataloader_workers=num_dataloader_workers,
            shuffle_buffer_batches=shuffle_buffer_batches,
            shuffle=shuffle,
            seed=seed,
        )

        self.log_stddev = log_stddev
        self.log_mean = log_mean

    def build_pipeline(self, jobs: "Iterable[int]"):
        builder = PipelineBuilder().add_source(jobs)
        if self.shuffle:
            builder = (
                builder
                .aggregate(self.shuffle_buffer_size, drop_last=False)
                .pipe(self.shuffle_examples, concurrency=2, name="shuffle_examples"))
        return (builder
                .pipe(self.read_example, concurrency=self.num_io_workers, name="read_chunk")
                .pipe(self.normalize, concurrency=self.num_cpu_workers, name="normalize")
                .aggregate(self.batch_size, drop_last=True)
                .pipe(self.batchify, concurrency=1, name="batchify")
                .add_sink(self.sink_buffer_size)
                .build(num_threads=self.num_threads))

    def read_example(self, job_id: int) -> "np.ndarray | None":
        res = self.index[job_id]
        if len(res) == 0:
            return None
        interval: "Interval" = next(iter(res))
        start_idx_this_file = interval.begin
        end_idx_this_file = interval.end
        filepath = self.data_root / "content" / interval.data
        example_start_idx = job_id - start_idx_this_file
        assert 0 <= example_start_idx < end_idx_this_file
        data = np.load(filepath, mmap_mode="r")
        data = data[example_start_idx]  # (H, W)
        return data[np.newaxis]  # (C=1, H, W)

    def normalize(self, example: "np.ndarray") -> "np.ndarray | None":
        x = np.clip(example, 1e-10, None)
        x = np.log(x)
        return (x - self.log_mean) / self.log_stddev

    @staticmethod
    def batchify(batch: list[np.ndarray]) -> np.ndarray:
        y = torch.utils.data.default_collate(batch)  # (B, C=1, H, W)
        assert y.ndim == 4 and y.shape[1] == 1
        return y

    def estimate_dataset_size(self, index_array):
        world_size: int = self.world_size
        batch_size: int = self.batch_size
        num_workers: int = self.num_dataloader_workers

        size = np.zeros((world_size, num_workers), dtype=np.int32)
        for rank in range(world_size):
            for worker in range(num_workers):
                for start, end, _ in index_array:
                    job_ids = range(start, end)[rank::world_size]  # 根据 rank 进行数据分片
                    job_ids = job_ids[worker::num_workers]
                    size[rank, worker] += len(job_ids)
        size = size.ravel()
        return np.min(size) // batch_size


class SPDLAstroImageDatasetStats(SPDLAstroImageDatasetBase):
    def __init__(
            self,
            data_root: str | os.PathLike,
            split: 'Literal["train", "valid", "test"]',
            preprocessor: str = "stats-preprocessor.pt",
            batch_size: int = 128,
            num_threads: int = 64,
            num_cpu_workers: int = 8,
            num_io_workers: int = 16,
            disable_ivar: bool = False,
            sink_buffer_size: int = 4,
            num_dataloader_workers: int = 0,
            shuffle_buffer_batches: int = 4,
            shuffle: bool = False,
            seed: int = 42,
    ) -> None:
        p = AstroImagePreprocessorStats.from_pretrained(Path(data_root) / preprocessor)

        super().__init__(
            data_root=data_root,
            split=split,
            batch_size=batch_size,
            num_threads=num_threads,
            num_cpu_workers=num_cpu_workers,
            num_io_workers=num_io_workers,
            disable_ivar=disable_ivar,
            crop_size=p.cropper.crop_size,
            sink_buffer_size=sink_buffer_size,
            num_dataloader_workers=num_dataloader_workers,
            shuffle_buffer_batches=shuffle_buffer_batches,
            shuffle=shuffle,
            seed=seed,
        )

        self.preprocessor = p

    def build_pipeline(self, job_ids):
        builder = (
            PipelineBuilder()
            .add_source(job_ids)
            .pipe(self.read_chunk, concurrency=self.num_io_workers, name="read_chunk")
            .pipe(self.preprocessor.forward, concurrency=self.num_cpu_workers, name="preprocess")
            .pipe(self.unbatch, concurrency=1, name="unbatch")
        )
        if self.shuffle:
            builder = (
                builder
                .aggregate(self.shuffle_buffer_size)
                .pipe(self.shuffle_examples, concurrency=1, name="shuffle_examples")
            )
        return (
            builder
            .aggregate(self.batch_size, drop_last=True)
            .pipe(self.batchify, concurrency=4, name="batchify")
            .add_sink(self.sink_buffer_size)
            .build(num_threads=self.num_threads)
        )
