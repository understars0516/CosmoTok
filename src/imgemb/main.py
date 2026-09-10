import math
import os
import typing
from pathlib import Path

import h5py
import numpy as np
import torch
from aion.codecs import CodecManager
from aion.model import AION
from astro_utils.helper import center_crop, inference_only
from imgtok.common import ImageDatasetAttrs
from jsonargparse import auto_cli
from qdrant_client import QdrantClient, models
from torch import nn
from tqdm import tqdm

if typing.TYPE_CHECKING:
    from typing import Any, Optional

    from aion.codecs.image import Image
    from jsonargparse.typing import Path_drw, Path_fr


@inference_only
class EmbeddingModel(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.tokenizer = CodecManager(device=device)
        self.encoder = AION.from_pretrained("polymathic-ai/aion-base").to(device)
        self.eval()

    def forward(self, images):
        tokens = self.tokenizer.encode(images)
        x = self.encoder.encode(tokens, num_encoder_tokens=600)
        return torch.mean(x, dim=1)


class EmbeddingServer:
    def __init__(self, name: str, url: str = None, embedding_dim: int = 768):
        if url is None:
            url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(url=url)
        self.name = name

    def create_collection(self):
        if not self.client.collection_exists(collection_name=self.name):
            self.client.create_collection(
                collection_name=self.name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )

    def upload_embeddings(self, images: "np.ndarray", payload: "list[dict[str, str]]" = None, ids: "list[int]" = None):
        self.client.upload_collection(
            collection_name=self.name,
            vectors=images,
            ids=ids,
            payload=payload,
        )


def embed_dir(data_root: "Path_drw", batch_size: int = 32, embedding_db: str = "multi_survey_images"):
    """递归地读取给定目录下的所有 HDF5 文件，将其中的图片转化为 embedding

    Args:
        data_root: 读取图像数据的根目录
        batch_size: 每次读取图像数
        embedding_db: embedding 数据库名称

    Returns:

    """
    data_root = Path(data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    model = EmbeddingModel(device)
    embedding_db = EmbeddingServer(embedding_db)
    embedding_db.create_collection()

    num_files = 0
    num_images = 0
    for dirpath, _, filenames in data_root.walk(follow_symlinks=True):
        for filename in filenames:
            if filename.endswith(".hdf5"):
                full_path = dirpath / filename
                num_images += _embed_one_file(model, embedding_db, full_path, batch_size)
                num_files += 1
    print(f"Embeds {num_files} files with {num_images} images")


def embed_file(filepath: "Path_fr", batch_size: int = 32, embedding_db: str = "multi_survey_images"):
    """对给定的 HDF5 文件中的图像, 计算它们的 embedding 向量, 并保存到向量数据库中

    Args:
        filepath: HDF5 文件路径
        batch_size: 批量处理图片数
        embedding_db: embedding 数据库名称

    Returns: None

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    model = EmbeddingModel(device)
    embedding_db = EmbeddingServer(embedding_db)
    embedding_db.create_collection()

    num_imgs = _embed_one_file(model, embedding_db, filepath, batch_size)
    print(f"Embeds 1 file with {num_imgs} images")


def _embed_one_file(embedding_model, embedding_db, filepath: str | os.PathLike, batch_size: int = 32) -> int:
    filepath = Path(filepath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    start_idx = 0
    num_images = 0
    with h5py.File(filepath, "r") as fp:
        total = fp[ImageDatasetAttrs.KEY_FLUX].shape[0]
        with tqdm(total=total, desc=f"Processing {filepath.name}") as pbar:
            while (batch := read_next_batch(fp, start_idx, batch_size, device)) is not None:
                img, payload, next_start_idx = batch
                embeddings = embedding_model(img).detach().cpu().numpy()
                assert embeddings.shape[0] == len(payload)
                num_images += embeddings.shape[0]
                embedding_db.upload_embeddings(embeddings, payload=payload)
                pbar.update(next_start_idx - start_idx)
                start_idx = next_start_idx
    return num_images


def read_next_batch(
        h5_group: h5py.Group,
        start_idx: int,
        batch_size: int,
        device: str,
        crop_size: int = 96,
) -> "Optional[tuple[Image, list[dict[str, Any]], int]]":
    filename = h5_group.file.filename
    id_prefix = filename.rsplit(".", maxsplit=1)[0]
    id_prefix = os.path.split(id_prefix)[-1]

    bands = [x.upper().decode() for x in h5_group[ImageDatasetAttrs.KEY_BAND][0]]
    attrs = ImageDatasetAttrs.get_concrete_attrs(bands)
    flux_ds = h5_group[attrs.KEY_FLUX]
    chunk_size = flux_ds.chunks[0]
    ds_size = flux_ds.shape[0]
    assert start_idx % chunk_size == 0, f"start_idx ({start_idx}) % chunk_size ({chunk_size}) != 0"
    batch_size = max(chunk_size, math.ceil(batch_size / chunk_size) * chunk_size)

    while start_idx < ds_size:
        end_idx = start_idx + batch_size

        indicator = h5_group[attrs.KEY_FILTER][start_idx:end_idx]
        valid_indices = attrs.validate(indicator)
        if (num_valid := len(valid_indices)) == 0:
            start_idx += batch_size
            continue

        flux = center_crop(flux_ds, crop_size, slice(start_idx, end_idx))
        number = np.arange(start_idx, end_idx)[:flux.shape[0]]
        if num_valid < flux.shape[0]:
            flux = flux[valid_indices]
            number = number[valid_indices]
        assert len(number) == len(flux), (
            f"{len(number)=} != {len(flux)=}, in file {filename}[{start_idx}:{end_idx})")

        payload = [{
            "number": int(i),
            "file": id_prefix,
            "survey": attrs.SURVEY_NAME,
        } for i in number]

        flux = torch.from_numpy(flux.astype("float32")).to(device)
        next_start_idx = start_idx + batch_size
        if next_start_idx >= ds_size:
            return None
        return attrs.create_example(flux, bands), payload, next_start_idx
    return None


if __name__ == "__main__":
    auto_cli({
        "dir": embed_dir,
        "file": embed_file,
    }, as_positional=False)
