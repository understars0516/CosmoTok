import os
import typing
from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np
import torch
from aion.codecs.image import Image
from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
from aion.codecs.preprocessing.image import CenterCrop, Clamp, RescaleToLegacySurvey
from aion.codecs.preprocessing.image import ImagePadder as OriginalImagePadder
from astro_utils.helper import maybe_slice, merge_dict_without_duplicates
from astro_utils.stats import AstroImageStats, Histogram
from imgtok.common import ImageDatum, get_survey
from torch import Tensor, nn

if typing.TYPE_CHECKING:
    from torch import Tensor


class Preprocessor(nn.Module, ABC):
    """Abstract base class for image pre-processing.

    所有预处理器都提供至少四个方法：

    * `transform()`: 用于推理时的正向预处理。
    * `inverse()`: 推理期逆向预处理。即将模型预测的数据，还原为预处理前的状态。该函数为 `transform()` 逆运算，但不强求数学上的可逆。
      因为常用的一些预处理运算，本身就是不可逆的，例如 `np.clip()`.
    * `forward()`: 用于训练的正向预处理。 与 `transform()` 不同，它可能会预处理更多字段的数据，例如对输入数据的 mask、权重 进行处理等。
      而这些字段在推理期可能是不存在的。
    * `backward()`: `forward()` 的逆过程。在训练期，通常用不到。默认将 `inverse()` 的定义为在 `torch.inference_mode()`
      下执行 `backward()`，因此，可以通过在子类中实现 `backward()` 方法，获得默认的 `inverse()` 方法定义。
    """
    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        with torch.inference_mode():
            return self.forward(*args, **kwargs)

    def inverse(self, *args, **kwargs):
        with torch.inference_mode():
            return self.backward(*args, **kwargs)


class LogNormPreprocessing(Preprocessor):
    """Perform pixel normalization and data type conversion.
    Ideal for images with a log-normal pixel distribution.

    Args:
        log_mean: The logarithmic mean of the pixel values (log(scale) of scipy.stats.lognorm).
        log_std: The logarithmic standard deviation of the pixel values.
    """

    def __init__(self, log_mean: "float | Tensor", log_std: "float | Tensor") -> None:
        super().__init__()
        self.register_buffer("log_mean", torch.as_tensor(log_mean, dtype=torch.float32))
        self.register_buffer("log_std", torch.as_tensor(log_std, dtype=torch.float32))

    def forward(self, x: "Tensor") -> "Tensor":
        """Normalize the given image batch.

        Args:
            x: The input image batch. Pixel values are expected to be
                in the range of ``[0, +inf]``.
        Returns:
            The normalized image batch.
        """
        x = torch.clip(x, 1e-10)
        x = torch.log(x)
        return (x - self.log_mean) / self.log_std

    def backward(self, x: "Tensor"):
        return torch.exp(x * self.log_std + self.log_mean)


class LogNormCDFTransform(Preprocessor):
    def __init__(self, log_mean: "float | Tensor", log_std: "float | Tensor") -> None:
        super().__init__()
        self.register_buffer("log_mean", torch.as_tensor(log_mean, dtype=torch.float32))
        self.register_buffer("log_std", torch.as_tensor(log_std, dtype=torch.float32))

        self.dist = None

    def forward(self, x: "Tensor") -> "Tensor":
        """Normalize the given image batch.

        Args:
            x: The input image batch. Pixel values are expected to be
        """
        if self.dist is None:
            self.dist = torch.distributions.log_normal.LogNormal(self.log_mean, self.log_std)

        return self.dist.cdf(x)

    def backward(self, x: "Tensor") -> "Tensor":
        if self.dist is None:
            self.dist = torch.distributions.log_normal.LogNormal(self.log_mean, self.log_std)

        return self.dist.icdf(x)


class AstroImagePreprocessorStats(Preprocessor):
    """Image pre-preprocessor based on statistics observed from the data.
    """

    def __init__(
            self,
            flux_log_means: "dict[str, float | Tensor]" = None,
            flux_log_stds: "dict[str, float | Tensor]" = None,
            ivar_log_means: "dict[str, float | Tensor]" = None,
            ivar_log_stds: "dict[str, float | Tensor]" = None,
            crop_size: int = 96,
            alpha: float = 1.0 / 3.0,
            beta: float = 3.0,
            eps: float = 1e-8,
    ) -> None:
        """
        Args:
            flux_log_means: The logarithmic mean of the pixel values for each band (log(scale) of scipy.stats.lognorm).
            flux_log_stds: The logarithmic standard deviation  for each band  of the pixel values.
        """
        super().__init__()
        # 默认可以为空，以方便从保存的模型中恢复
        flux_log_means = flux_log_means or {}
        flux_log_stds = flux_log_stds or {}
        ivar_log_means = ivar_log_means or {}
        ivar_log_stds = ivar_log_stds or {}

        nbands = len(BAND_TO_INDEX)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.cropper = CenterCrop(crop_size)
        self.channel_padder = ImagePadder()

        self.survey_band_index = {
            "HSC": maybe_slice([i for band, i in BAND_TO_INDEX.items() if band.startswith("HSC")]),
            "DES": maybe_slice([i for band, i in BAND_TO_INDEX.items() if band.startswith("DES")]),
        }

        hsc_slice = self.survey_band_index["HSC"]
        des_slice = self.survey_band_index["DES"]

        # 要能够向量化计算, 必须保证两个 Survey 的索引是分别连续的.
        assert isinstance(hsc_slice, slice) and hsc_slice.step == 1
        assert isinstance(des_slice, slice) and des_slice.step == 1

        flux_log_means_pt = torch.zeros((nbands, 1, 1), dtype=torch.float32)
        flux_log_stds_pt = torch.ones((nbands, 1, 1), dtype=torch.float32)
        ivar_log_means_pt = torch.zeros((nbands, 1, 1), dtype=torch.float32)
        ivar_log_stds_pt = torch.ones((nbands, 1, 1), dtype=torch.float32)

        for i, (band, band_index) in enumerate(BAND_TO_INDEX.items()):
            assert i == band_index
            flux_log_means_pt[i] = flux_log_means.get(band, 0.)
            flux_log_stds_pt[i] = flux_log_stds.get(band, 1.)
            ivar_log_means_pt[i] = ivar_log_means.get(band, 0.)
            ivar_log_stds_pt[i] = ivar_log_stds.get(band, 1.)

        # noinspection PyTypeChecker
        self.flux_processors = nn.ModuleDict([
            ("HSC", LogNormPreprocessing(flux_log_means_pt[hsc_slice], flux_log_stds_pt[hsc_slice])),
            ("DES", LogNormPreprocessing(flux_log_means_pt[des_slice], flux_log_stds_pt[des_slice])),
        ])
        self.flux_compressor = lambda x: beta * torch.tanh(alpha * x)

        # noinspection PyTypeChecker
        self.ivar_processors = nn.ModuleDict([
            ("HSC", LogNormCDFTransform(ivar_log_means_pt[hsc_slice], ivar_log_stds_pt[hsc_slice])),
            ("DES", LogNormCDFTransform(ivar_log_means_pt[des_slice], ivar_log_stds_pt[des_slice])),
        ])

    @classmethod
    def from_pretrained(cls, model_path: str | os.PathLike):
        model = cls()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_stats(cls, stats: "list[AstroImageStats]"):
        def collect_fit_params(sketches):
            log_means = {}
            log_stds = {}
            for band, sketch in sketches.items():
                hist = Histogram.from_sketch(sketch)
                sigma, scale = hist.fit_lognorm()
                mu = np.log(scale)
                log_means[band] = mu
                log_stds[band] = sigma
            return log_means, log_stds

        flux_log_means = {}
        flux_log_stds = {}
        ivar_log_means = {}
        ivar_log_stds = {}

        for s in stats:
            means, stds = collect_fit_params(s.flux_sketches)
            merge_dict_without_duplicates(flux_log_means, means)
            merge_dict_without_duplicates(flux_log_stds, stds)

            means, stds = collect_fit_params(s.ivar_sketches)
            merge_dict_without_duplicates(ivar_log_means, means)
            merge_dict_without_duplicates(ivar_log_stds, stds)

        return cls(flux_log_means, flux_log_stds, ivar_log_means, ivar_log_stds)

    def forward(self, x: "ImageDatum") -> "ImageDatum":
        survey = x.get_survey()

        x = self.standardize(x)
        if x.ivar is not None:
            (flux, ivar, mask), channel_mask = self.channel_padder.forward((x.flux, x.ivar, x.mask), x.bands)
        else:
            ivar = None
            (flux, mask), channel_mask = self.channel_padder.forward((x.flux, x.mask), x.bands)

        s = self.survey_band_index[survey]
        flux[:, s] = self.flux_compressor(self.flux_processors[survey].forward(flux[:, s]))
        if ivar is not None:
            ivar[:, s] = self.ivar_processors[survey].forward(ivar[:, s])

        return ImageDatum(flux, ivar, mask, channel_mask, x.bands)

    def transform(self, x: "Image") -> "tuple[Tensor, Tensor]":
        survey = get_survey(x.bands)

        with torch.inference_mode():
            flux = self.cropper(x.flux)  # (B=1, C=5, H=96, W=96)
            flux, channel_mask = self.channel_padder.forward(flux, x.bands)

            s = self.survey_band_index[survey]
            flux[:, s] = self.flux_compressor(self.flux_processors[survey].forward(flux[:, s]))

            return flux, channel_mask

    def backward(self, x: "Tensor", bands: list[str]) -> "Image":
        survey = get_survey(bands)

        s = self.survey_band_index[survey]
        x[:, s] = self.flux_processors[survey].backward(self.inverse_compress(x[:, s]))

        return Image(self.channel_padder.backward(x, bands), bands)

    def standardize(self, x: ImageDatum):
        """ 将数据的类型和形状标准化

        :param x: 输入的 ImageDatum 数据
        :return: 标准化的 ImageDatum 类型数据
        """
        mask = x.mask
        if mask.ndim == 3:
            mask = mask[:, np.newaxis, ...]  # (B=1, C=1, H=160, W=160)
            mask = np.broadcast_to(mask, x.flux.shape)  # (B=1, C=5, H, W)

        flux = torch.from_numpy(x.flux)
        mask = torch.from_numpy(mask)
        ivar = torch.from_numpy(x.ivar) if x.ivar is not None else None

        flux = self.cropper(flux)  # (B=1, C=5, H=96, W=96)
        mask = self.cropper(mask)  # (B=1, C=5, H=96, W=96)
        ivar = self.cropper(ivar) if ivar is not None else None
        return ImageDatum(flux, ivar, mask, x.channel_mask, x.bands)

    # 定义逆变换
    def inverse_compress(self, y):
        """
        逆变换: x = (1/alpha) * atanh(y/beta)
        """
        alpha = self.alpha
        beta = self.beta
        # 确保输入在 (-beta, beta) 内
        y_clipped = torch.clamp(y, -beta + self.eps, beta - self.eps)
        return torch.atanh(y_clipped / beta) / alpha


class RangeCompressor:
    def __init__(
            self,
            range_compression_factor: float = 0.01,
            mult_factor: float = 10.0,
    ):
        self.range_compression_factor = range_compression_factor
        self.mult_factor = mult_factor

    def forward(self, x: "Tensor") -> "Tensor":
        return self.mult_factor * (torch.arcsinh(x / self.range_compression_factor) * self.range_compression_factor)

    def backward(self, x: "Tensor") -> "Tensor":
        x = x / self.mult_factor
        return torch.sinh(x / self.range_compression_factor) * self.range_compression_factor


class ImagePadder(OriginalImagePadder):
    def forward(self, image: "Union[Tensor, Sequence[Tensor]]", bands: list[str]):
        # 相对于基类版本，提供了一次性 pad 多个图像的能力。

        # Check if bands are valid
        self._check_bands(bands)

        # Create a list of new channel indices based on the order of bands
        new_channel_indices = [BAND_TO_INDEX[band] for band in bands if band in BAND_TO_INDEX]
        num_channels = self.nbands

        if isinstance(image, (list, tuple)):
            device = image[0].device
            batch_size = image[0].shape[0]
            result = []
            for img in image:
                padded_image = self.create_channel_padded_tensor(img, num_channels, new_channel_indices)
                result.append(padded_image)
        else:
            device = image.device
            batch_size = image.shape[0]
            result = self.create_channel_padded_tensor(image, num_channels, new_channel_indices)

        # Get boolean mask of channels that are present
        channel_mask = torch.zeros(num_channels, dtype=torch.bool).to(device)
        channel_mask[new_channel_indices] = True
        channel_mask = channel_mask.unsqueeze(0).expand(batch_size, -1)

        return result, channel_mask

    @staticmethod
    def create_channel_padded_tensor(image: "Tensor", channels, indices):
        batch, _, height, width = image.shape
        # Create a new image array with the correct number of channels
        padded_image = torch.zeros((batch, channels, height, width), dtype=image.dtype, device=image.device)

        # Vectorized assignment of the original channels to the new positions
        padded_image[:, indices, :, :] = image[:, :len(indices), :, :]

        return padded_image


class AstroImagePreprocessorAdHoc:
    def __init__(
            self,
            # p_values: "PValues",
            # expected_rescaled_ivar: "Sequence[float]" = (0.8, 0.95, 0.99, 1.0),
            range_compression_factor: float = 0.01,
            mult_factor: float = 10.0,
            crop_size: int = 96,
    ) -> None:
        self.clamper = Clamp()
        self.cropper = CenterCrop(crop_size)
        self.rescaler = RescaleToLegacySurvey()
        self.range_compressor = RangeCompressor(range_compression_factor, mult_factor)
        self.image_padder = ImagePadder()

        # self.p_values = np.array([0.0, *p_values], dtype=np.float32)
        # self.expected_rescaled_ivar = np.array((0.0, *expected_rescaled_ivar), dtype=np.float32)

    def __call__(self, x: ImageDatum):
        with torch.inference_mode():
            x = self.standardize(x)
            x = self.normalize_ivar(x)
            x = self.clamp(x)
            x = self.rescale(x)
            x = self.range_compress(x)
            x = self.pad_image_channels(x)
            return self.squeeze(x)

    # noinspection PyMethodMayBeStatic
    def ivar_transform(self, x: "np.ndarray") -> "np.ndarray":
        # # if self._ivar_transform is None:
        # #     self._initialize_transforms()
        # x = np.clip(x, 0.0, self.p_values[-1])
        # shape = x.shape
        # # y = self._ivar_transform(x.ravel())
        # y = np.interp(x.ravel(), self.p_values, self.expected_rescaled_ivar)
        # return y.reshape(shape)
        return x

    def standardize(self, x: ImageDatum):
        """ 将数据的类型和形状标准化

        :param x: 输入的 ImageDatum 数据
        :return: 标准化的 ImageDatum 类型数据
        """
        mask = x.mask
        if mask.ndim == 3:
            mask = mask[:, np.newaxis, ...]  # (B=1, C=1, H=160, W=160)
            mask = np.broadcast_to(mask, x.flux.shape)  # (B=1, C=5, H, W)

        flux = torch.tensor(x.flux, device="cpu")
        ivar = torch.tensor(x.ivar, device="cpu")
        mask = torch.tensor(mask, device="cpu")

        flux = self.cropper(flux)  # (B=1, C=5, H=96, W=96)
        ivar = self.cropper(ivar)  # (B=1, C=5, H=96, W=96)
        mask = self.cropper(mask)  # (B=1, C=5, H=96, W=96)
        return ImageDatum(flux, ivar, mask, x.channel_mask, x.bands)

    def normalize_ivar(self, x: ImageDatum):
        x.ivar = self.ivar_transform(x.ivar)
        return x

    # @staticmethod
    # def to_tensor(x: ImageDatum):
    #     flux = torch.tensor(x.flux, device="cpu")
    #     ivar = torch.tensor(x.ivar, device="cpu")
    #     mask = torch.tensor(x.mask, device="cpu")
    #     return ImageDatum(flux, ivar, mask, None)

    # def crop(self, x: ImageDatum):
    #     flux = self.cropper(x.flux)  # (B=1, C=5, H=96, W=96)
    #     ivar = self.cropper(x.ivar)  # (B=1, C=5, H=96, W=96)
    #     mask = self.cropper(x.mask)  # (B=1, C=5, H=96, W=96)
    #     return ImageDatum(flux, ivar, mask, None)

    def clamp(self, x: ImageDatum):
        x.flux = self.clamper(x.flux, x.bands)
        return x

    def rescale(self, x: ImageDatum):
        x.flux = self.rescaler.forward(x.flux, x.get_survey())
        return x

    def range_compress(self, x: ImageDatum):
        x.flux = self.range_compressor.forward(x.flux)
        return x

    def pad_image_channels(self, x: ImageDatum):
        (flux, ivar, mask), channel_mask = self.image_padder.forward((x.flux, x.ivar, x.mask), x.bands)
        return ImageDatum(flux, ivar, mask, channel_mask, x.bands)

    @staticmethod
    def squeeze(x: ImageDatum):
        x.flux = x.flux.squeeze(0)  # (C=9, H=96, W=96)
        x.ivar = x.ivar.squeeze(0)  # (C=9, H=96, W=96)
        x.mask = x.mask.squeeze(0)  # (C=9, H=96, W=96)
        x.channel_mask = x.channel_mask.squeeze(0)
        return x
