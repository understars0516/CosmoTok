import typing

import numpy as np
import torch
from aion.codecs.image import AutoencoderImageCodec, Image, SubsampledLinear
from aion.codecs.modules.magvit import MagVitAE
from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
from aion.codecs.quantizers import FiniteScalarQuantizer as FSQuantizer
from aion.codecs.utils import CodecPytorchHubMixin
from imgtok.model.prepare import AstroImagePreprocessorStats, LogNormPreprocessing
from torch import nn

if typing.TYPE_CHECKING:
    from os import PathLike
    from typing import Sequence, Union

    from torch import Tensor


class FiniteScalarQuantizer(FSQuantizer):
    def __init__(
            self,
            levels: list[int],
            eps: float = 1e-3,
    ):
        super().__init__(levels, eps)
        # 没必要设置为 buffer, 这会带来更多通信开销
        # self.register_buffer("no_quant_prob", torch.tensor(1.0), persistent=False)
        self.no_quant_prob = 0.0

    def _quantize(
        self, z: "Tensor",  # Float[torch.Tensor, " B L d"]
    ) -> "Tensor":          # Float[torch.Tensor, " B L d"]
        def round_ste(z):
            zhat = z.round()
            quantized_ = z + (zhat - z).detach()  #
            if self.training and self.no_quant_prob > 0.0:
                with torch.no_grad():
                    no_quant_prob_shape = (z.shape[0], ) + (1, ) * (z.ndim - 1)
                    m = torch.empty(no_quant_prob_shape, device=z.device, dtype=z.dtype).bernoulli_(self.no_quant_prob)
                return z * m + quantized_ * (1.0 - m)
            else:
                return quantized_

        quantized = round_ste(self._bound(z))
        # Renormalize to [-1, 1].
        half_width = self.levels // 2
        return quantized / half_width

    def set_no_quant_prob(self, no_quant_prob: float) -> float:
        val = np.clip(no_quant_prob, 0.0, 1.0)
        # self.no_quant_prob.fill_(val)
        self.no_quant_prob = val
        return val

    def get_no_quant_prob(self) -> float:
        return self.no_quant_prob


class ImageCodec(AutoencoderImageCodec, CodecPytorchHubMixin):
    def __init__(
            self,
            quantizer_levels: "Sequence[int]" = (7, 5, 5, 5, 5),
            hidden_dims: int = 512,
            embedding_dim: int = 5,
            multisurvey_projection_dims: int = 54,
            n_compressions: int = 2,
            num_consecutive: int = 4,
            range_compression_factor: float = 0.01,
            mult_factor: float = 10.0,
    ):
        """
        MagViT Autoencoder for images.

        Args:
            quantizer_levels: Levels for the FiniteScalarQuantizer.
            hidden_dims: Number of hidden dimensions in the network.
            n_compressions: Number of compressions in the network.
            num_consecutive: Number of consecutive residual layers per compression.
            embedding_dim: Dimension of the latent space.
            range_compression_factor: Range compression factor.
            mult_factor: Multiplication factor.
        """
        model = MagVitAE(
            n_bands=multisurvey_projection_dims,
            hidden_dims=hidden_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        # noinspection PyTypeChecker
        quantizer = FiniteScalarQuantizer(levels=quantizer_levels)
        # noinspection PyTypeChecker
        super().__init__(
            quantizer,
            model.encode,
            model.decode,
            hidden_dims,
            embedding_dim,
            multisurvey_projection_dims,
            range_compression_factor,
            mult_factor,
        )
        self.model = model

    def forward(self, x: "Tensor", channel_mask: "Tensor") -> "Tensor":
        x = self.subsample_in(x, channel_mask)
        h = self.encoder(x)
        h = self.pre_quant_proj(h)
        batch_size, channels, *spatial_size = h.shape
        h = h.reshape(batch_size, channels, -1)

        z = self.quantizer.quantize(h)

        # z is flattened, need to reshape
        batch_size, embedding_dim, _ = z.shape
        z = z.reshape(batch_size, embedding_dim, *spatial_size)
        h = self.post_quant_proj(z)
        decoded_flux_raw = self.decoder(h)

        full_dim_channel_mask = torch.ones(
            (z.shape[0], self.image_padder.nbands), device=z.device, dtype=torch.bool
        )
        return self.subsample_out(decoded_flux_raw, full_dim_channel_mask)

    def reconstruct(self, img: "Image") -> "Image":
        embeddings = self._encode(img)
        quant_embeddings = self.quantizer.quantize(embeddings)
        return self._decode(quant_embeddings, bands=img.bands)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
    ):
        """Load a codec model from a pretrained model repository.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                model repository.
            *model_args: Additional positional arguments to pass to the model
                constructor.
            **kwargs: Additional keyword arguments to pass to the model
                constructor.

        Returns:
            The loaded codec model.

        Raises:
            ValueError: If the class is not a codec subclass or modality is invalid.
        """
        return super().from_pretrained(pretrained_model_name_or_path, Image, *model_args, **kwargs)

    @staticmethod
    def _validate_codec_modality(codec, modality):
        return True


class AutoencoderCosmoGridImageCodec(nn.Module):
    """Autoencoder for CosmoGrid images."""

    def __init__(
            self,
            hidden_dims: int = 256,
            embedding_dim: int = 5,
            downsampled_channel_dim: int = 16,
            quantizer_levels: "Sequence[int]" = (7, 7, 5, 5, 5),
            n_compressions: int = 2,
            num_consecutive: int = 4,
            log_mean: float = -4.9179,
            log_std: float = 0.4947,
    ):
        super().__init__()
        if len(quantizer_levels) != embedding_dim:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must match len(quantizer_levels) ({len(quantizer_levels)}): "
                f"{list(quantizer_levels)}"
            )
        self.preprocessor = LogNormPreprocessing(log_mean, log_std)

        # noinspection PyTypeChecker
        self.quantizer = FiniteScalarQuantizer(levels=quantizer_levels)

        self.downsample = nn.Conv2d(
            in_channels=1,
            out_channels=downsampled_channel_dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.ae = MagVitAE(
            n_bands=downsampled_channel_dim,
            hidden_dims=hidden_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )

        self.upsample = nn.ConvTranspose2d(
            in_channels=downsampled_channel_dim,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        # Go down to size of levels
        self.pre_quant_proj = nn.Conv2d(
            hidden_dims, embedding_dim, kernel_size=1, stride=1, padding=0
        )
        # Go back to the original size
        self.post_quant_proj = nn.Conv2d(
            embedding_dim, hidden_dims, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: "Tensor"):
        """Module 的 forward 方法定义. 该函数用于构建训练时的计算图.

        :param x: 输入图像
        :return: 重构的图像
        """
        x = self.downsample(x)
        x = self.ae.encode(x)
        h = self.pre_quant_proj(x)
        if h.shape[1] != self.quantizer.embedding_dim:
            raise RuntimeError(
                f"Quantizer embedding_dim mismatch: pre_quant_proj out_channels={h.shape[1]}, "
                f"quantizer.embedding_dim={self.quantizer.embedding_dim}, "
                f"quantizer_levels={self.quantizer.levels.tolist()}"
            )
        z = self.quantizer.quantize(h)
        h = self.post_quant_proj(z)
        x = self.ae.decode(h)
        return self.upsample(x)

    def _encode(self, x: "Tensor") -> "Tensor":
        """该函数负责将原始数据(未预处理)编码为连续隐变量(量化前的隐变量).

        :param x: 原始数据
        :return: 隐变量
        """
        x = self.preprocessor(x)
        x = self.downsample(x)
        x = self.ae.encode(x)
        h = self.pre_quant_proj(x)
        return torch.flatten(h, 2)

    def _decode(self, z: "Tensor") -> "Tensor":
        # z is flattened, need to reshape
        batch_size, embedding_dim, n_tokens = z.shape
        spatial_size = int(n_tokens**0.5)
        z = z.reshape(batch_size, embedding_dim, spatial_size, spatial_size)

        h = self.post_quant_proj(z)
        x = self.ae.decode(h)
        x = self.upsample(x)
        return self.preprocessor.inverse(x)

    def encode(self, x: "Tensor"):
        x = self._encode(x)
        return self.quantizer.encode(x)

    def decode(self, codes: "Tensor") -> "Tensor":
        """
        Decodes the given latent tensor `z` back into an Image object.

        Args:
            codes: The latent tensor to decode.
        Returns:
            An Image tensor.
        """
        z = self.quantizer.decode(codes)
        return self._decode(z)


class ImageCodecV2(nn.Module):
    def __init__(
            self,
            quantizer_levels: "Sequence[int]" = (7, 5, 5, 5, 5),
            hidden_dims: int = 512,
            embedding_dim: int = 5,
            multisurvey_projection_dims: int = 54,
            n_compressions: int = 2,
            num_consecutive: int = 4,
            preprocessor: "Union[str, PathLike]" = None,
    ):
        """
        MagViT Autoencoder for images.

        Args:
            quantizer_levels: Levels for the FiniteScalarQuantizer.
            hidden_dims: Number of hidden dimensions in the network.
            n_compressions: Number of compressions in the network.
            num_consecutive: Number of consecutive residual layers per compression.
            embedding_dim: Dimension of the latent space.
        """
        super().__init__()

        if preprocessor is None:
            self.preprocessor = None
        else:
            self.preprocessor = AstroImagePreprocessorStats.from_pretrained(preprocessor)

        self.num_channels = len(BAND_TO_INDEX)
        self.subsample_in = SubsampledLinear(
            dim_in=self.num_channels,
            dim_out=multisurvey_projection_dims,
            subsample_in=True,
        )
        self.subsample_out = SubsampledLinear(
            dim_in=multisurvey_projection_dims,
            dim_out=self.num_channels,
            subsample_in=False,
        )
        # Go down to size of levels
        self.pre_quant_proj = nn.Conv2d(hidden_dims, embedding_dim, kernel_size=1, stride=1, padding=0)

        # Go back to the original size
        self.post_quant_proj = nn.Conv2d(embedding_dim, hidden_dims, kernel_size=1, stride=1, padding=0)

        self.ae = MagVitAE(
            n_bands=multisurvey_projection_dims,
            hidden_dims=hidden_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )

        # noinspection PyTypeChecker
        self.quantizer = FiniteScalarQuantizer(levels=quantizer_levels)

    def forward(self, x: "Tensor", channel_mask: "Tensor") -> "Tensor":
        h = self._nn_encode(x, channel_mask)
        z = self.quantizer.quantize(h)
        return self._nn_decode(z)

    def reconstruct(self, img: "Image") -> "Image":
        embeddings = self._encode(img)
        quant_embeddings = self.quantizer.quantize(embeddings)
        return self._decode(quant_embeddings, bands=img.bands)

    def encode(self, x: "Image") -> "Tensor":
        z = self._encode(x)
        return self.quantizer.encode(z)

    def decode(self, tokens, bands: list[str] = None) -> "Image":
        z = self.quantizer.decode(tokens)
        return self._decode(z, bands)

    def _encode(self, x: "Image") -> "Tensor":
        """该函数负责将原始数据(未预处理)编码为连续隐变量(量化前的隐变量).

        :param x: 原始数据
        :return: 隐变量
        """
        x, channel_mask = self.preprocessor.transform(x)
        return self._nn_encode(x, channel_mask)

    def _decode(self, z: "Tensor", bands: list[str] = None) -> "Image":
        x = self._nn_decode(z)
        return self.preprocessor.inverse(x, bands)

    def _nn_encode(self, x: "Tensor", channel_mask: "Tensor") -> "Tensor":
        x = self.subsample_in(x, channel_mask)
        x = self.ae.encode(x)
        h = self.pre_quant_proj(x)
        return torch.flatten(h, 2)

    def _nn_decode(self, z: "Tensor") -> "Tensor":
        batch_size, embedding_dim, n_tokens = z.shape
        spatial_size = int(n_tokens ** 0.5)
        z = z.reshape(batch_size, embedding_dim, spatial_size, spatial_size)
        h = self.post_quant_proj(z)
        x = self.ae.decode(h)
        full_dim_channel_mask = torch.ones((batch_size, self.num_channels), device=z.device, dtype=torch.bool)
        return self.subsample_out(x, full_dim_channel_mask)
