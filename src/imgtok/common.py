from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class ImageDatum:
    flux: "Union[Tensor, np.ndarray]"
    ivar: "Union[Tensor, np.ndarray]"
    mask: "Union[Tensor, np.ndarray]"
    channel_mask: "Optional[Union[Tensor, np.ndarray]]" = None
    bands: "Optional[list[str]]" = None

    @property
    def batch_size(self) -> int:
        return self.flux.shape[0] if self.flux.ndim == 4 else 0

    def __getitem__(self, item):
        if self.flux.ndim == 4 or (self.flux.ndim == 3 and item is None):
            flux = self.flux[item]
            ivar = self.ivar[item] if self.ivar is not None else None
            mask = self.mask[item]
            channel_mask = self.channel_mask[item] if self.channel_mask is not None else None
            # indicator = self.indicator[item] if self.indicator is not None else None
            # return ImageDatum(flux, ivar, mask, channel_mask, self.bands, indicator)
            return ImageDatum(flux, ivar, mask, channel_mask, self.bands)
        return self

    def get_survey(self) -> str:
        if self.bands is None:
            return "unknown"
        return self.bands[0].split("-")[0]

    def get_image_dataset_attrs(self):
        return ImageDatasetAttrs.get_concrete_attrs(self.bands)

    # def validate(self):
    #     if (c := self.get_image_dataset_attrs()) is None:
    #         return False
    #     elif self.indicator is not None:
    #         return c.validate(self.indicator)
    #     return False


class ImageDatasetAttrs(ABC):
    KEY_FILTER = None         # 用于筛选
    KEY_FLUX = "image_array"  # 图像数据 (Flux)
    KEY_IVAR = "image_ivar"   # 逆方差
    KEY_MASK = "image_mask"   # 掩码
    KEY_BAND = "image_band"   # 图像数据每个通道的band名字

    CHANNELS = None
    FILTER_THRESHOLD = None

    @classmethod
    @abstractmethod
    def validate(cls, indicator: "np.ndarray") -> "np.ndarray":
        pass

    @staticmethod
    def get_concrete_attrs(bands: list[str]):
        name = bands[0].split("-")[0]
        match name:
            case "HSC":
                return HSCImageDatasetAttrs
            case "DES":
                return LegacySurveyImageDatasetAttrs
            case _:
                raise ValueError(f"Unknown survey: {name}")


class HSCImageDatasetAttrs(ImageDatasetAttrs):
    KEY_FILTER = "i_cmodel_mag"  # 用于筛选

    CHANNELS = 5
    FILTER_THRESHOLD = 22.5
    CHUNK_SIZE = 1

    @classmethod
    def validate(cls, indicator_values):
        valid_mask = np.logical_and(indicator_values < cls.FILTER_THRESHOLD, np.isfinite(indicator_values))
        valid_idxes, = np.nonzero(valid_mask)
        return valid_idxes


class LegacySurveyImageDatasetAttrs(ImageDatasetAttrs):
    KEY_FILTER = "FLUX_Z"  # 用于筛选

    CHANNELS = 4
    FILTER_THRESHOLD = 3.98107
    CHUNK_SIZE = 25

    @classmethod
    def validate(cls, indicator_values):
        valid_mask = np.logical_and(indicator_values > cls.FILTER_THRESHOLD, np.isfinite(indicator_values))
        valid_idxes, = np.nonzero(valid_mask)
        return valid_idxes


def get_survey(bands: "list[str]") -> str:
    return bands[0].split("-")[0]
