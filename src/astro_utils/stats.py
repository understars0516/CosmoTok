import base64
import os
import pickle  # noqa: S403
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astro_utils.helper import is_sorted
from num2tex import num2tex
from scipy.stats import norm

if TYPE_CHECKING:
    from typing import Mapping, Sequence

    from ddsketch import DDSketch


@dataclass
class AstroImageStats:
    flux_sketches: "dict[str, DDSketch]"
    ivar_sketches: "dict[str, DDSketch]"
    masked_points: int  # 采样到的数据中, 被 mask 的点数数 (与像素不同, 不同的通道可能被分别 mask, 每个通道分别算作不同的点).
    total_points: int   # 采样数据包含的总点数
    total_samples: int  # 采样数据包含的总样本数
    valid_samples: int  # 数据中包含的有效(可用于训练) 的样本数.
    num_samples: int    # 数据中包含的总的样本数

    def merge(self, other: "AstroImageStats"):
        for key in self.flux_sketches:
            self.flux_sketches[key].merge(other.flux_sketches[key])
        for key in self.ivar_sketches:
            self.ivar_sketches[key].merge(other.ivar_sketches[key])
        self.masked_points += other.masked_points
        self.total_points += other.total_points
        self.total_samples += other.total_samples
        self.valid_samples += other.valid_samples
        self.num_samples += other.num_samples

    def save(self, path: str | os.PathLike):
        with open(path, "wb") as fp:
            # noinspection PyTypeChecker
            pickle.dump(self, fp)

    def _repr_markdown_(self):
        quantiles = [0.05, 0.2, 0.5, 0.8, 0.95]

        def generate_markdown_table(header, data, indent=0):
            """
            生成 Markdown 格式的表格字符串。
            :param header: 列表，表头
            :param data: 二维列表，行数据
            :param indent: 缩进长度
            """
            indent = " " * indent
            lines = []

            # 1. 构建表头
            header_line = "".join([indent, "| ", " | ".join(str(h) for h in header), " |"])
            lines.append(header_line)

            # 2. 构建分割线 (Markdown 必须有这一行才能识别为表格)
            separator = "".join([indent, "| ", " | ".join(["---"] * len(header)), " |"])
            lines.append(separator)

            # 3. 构建数据行
            for row in data:
                row_line = "".join([indent, "| ", " | ".join(str(cell) for cell in row), " |"])
                lines.append(row_line)

            return "\n".join(lines)

        def generate_image_data(fig):
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = base64.b64encode(buf.read())
            return img.decode("utf-8")

        def generate_image_html(img, name, indent=0):
            indent = " " * indent
            result = [
                f'{indent}<div style="text-align: center">',
                f'{indent}    <img src="data:image/png;base64,{img}" alt="{name}" align="center">',
                f"{indent}</div>"]
            return "\n".join(result)

        def collect_sketch_vis_data(sketches):
            table_header = ["", "sigma", "scale"]
            for quantile in quantiles:
                table_header.append(f"P{int(quantile * 100):02d}")
            table_data = []
            image_data = []

            for key in sketches:
                sketch = sketches[key]
                hist = Histogram.from_sketch(sketch)
                q_values = {q: sketch.get_quantile_value(q) for q in quantiles}
                fig, row_ = hist.plot(f"Histogram for {key} band", q_values)
                row = [key, *row_, *q_values.values()]
                row = [x if isinstance(x, str) else f"${num2tex(x):.3g}$" for x in row]
                table_data.append(row)
                image_data.append((key, generate_image_data(fig)))
                plt.close(fig)

            return (table_header, table_data), image_data

        basic_header = ["描述", "值"]
        basic_data = [
            ["数据集总的样本数", self.num_samples],
            ["数据集中有效的样本数", self.valid_samples],
            ["有效样本比例", f"{self.valid_samples / self.num_samples:.2%}"],
            ["采样数据数", self.total_samples],
            ["采样比例", f"{self.total_samples / self.num_samples:.2%}"],
            ["采样数据总点数", self.total_points],
            ["采样数据被掩码点数", self.masked_points],
            ["被掩码比例", f"{self.masked_points / self.total_points:.2%}"],
        ]

        flux_summary, flux_images = collect_sketch_vis_data(self.flux_sketches)
        ivar_summary, ivar_images = collect_sketch_vis_data(self.ivar_sketches)
        return "\n".join([
            "## 基本信息\n",
            f"{generate_markdown_table(basic_header, basic_data)}",
            "## 数据分布",
            "### Flux",
            "* 分布摘要",
            f"{generate_markdown_table(flux_summary[0], flux_summary[1], 2)}",
            "* 直方图",
            *[generate_image_html(img, name, 2) for name, img in flux_images],
            "### IVAR",
            "* 分布摘要",
            f"{generate_markdown_table(ivar_summary[0], ivar_summary[1], 2)}",
            "* 直方图",
            *[generate_image_html(img, name, 2) for name, img in ivar_images],
        ])


@dataclass
class Histogram:
    counts: np.ndarray
    bin_left_edges: np.ndarray
    alpha: float

    def __post_init__(self):
        assert is_sorted(self.bin_left_edges)

    # noinspection PyProtectedMember,PyUnresolvedReferences
    @classmethod
    def from_sketch(cls, sketch: "DDSketch"):
        # 从 sketch 中抽取统计信息
        counts = []
        bin_left_edges = []
        for i, count in enumerate(sketch._store.bins):
            bucket_idx = i + sketch._store.offset
            left_edge = sketch._mapping.value(bucket_idx)
            bin_left_edges.append(left_edge)
            counts.append(count)
        bin_left_edges = np.array(bin_left_edges)
        counts = np.array(counts)

        gamma = sketch._mapping.relative_accuracy
        alpha = (1 + gamma) / (1 - gamma)

        # 排序以确保绘图正确
        sorted_indices = np.argsort(bin_left_edges)
        bin_left_edges = bin_left_edges[sorted_indices]
        counts = counts[sorted_indices]
        return cls(counts, bin_left_edges, alpha)

    def fit_lognorm(self):
        counts = np.array(self.counts)
        left_edges = np.array(self.bin_left_edges)

        # 1. 修正重心偏移：使用几何中点
        # 右边界 = 左边界 * alpha
        bin_centers = left_edges * np.sqrt(self.alpha)

        # 2. 转换为对数空间进行加权计算
        log_centers = np.log(bin_centers)
        total_n = np.sum(counts)

        mu_log = np.sum(counts * log_centers) / total_n
        sigma_log = np.sqrt(np.sum(counts * (log_centers - mu_log) ** 2) / total_n)

        scale = np.exp(mu_log)

        return sigma_log, scale

    def plot(
            self,
            title: str,
            q_values: "Mapping[float, float]" = None,
            colors: "Sequence[str]" = ("blue", "orange", "red", "purple", "green"),
    ):
        widths = self.bin_left_edges * (self.alpha - 1)  # 计算每个桶的宽度（对数桶的宽度是不等的）

        # 使用 Matplotlib 绘制直方图
        fig, ax1 = plt.subplots(figsize=(10, 6))

        if self.bin_left_edges.size <= 0:
            return fig, (None, None), {}

        ax1.bar(self.bin_left_edges, self.counts, width=widths, align="edge",
                alpha=0.7, color="skyblue", edgecolor="navy", label="DDSketch Bins")

        # 如果有分位数数据, 则绘制它们
        if q_values:
            # 绘制分位数竖线
            for (q, val), color in zip(q_values.items(), colors, strict=False):
                ax1.axvline(val, color=color, linestyle="--", label=f"P{int(q * 100):02d}: {val:.2g}")

        # 绘制拟合的 LogNormal 分布
        sigma, scale = self.fit_lognorm()
        x_plot = np.logspace(np.log10(self.bin_left_edges.min()), np.log10(self.bin_left_edges.max()), 1000)
        y_plot = norm.pdf(np.log(x_plot), loc=np.log(scale), scale=sigma)

        ax2 = ax1.twinx()
        ax2.plot(x_plot, y_plot, color="crimson", lw=2, label="LogNormal PDF")
        ax2.set_ylim(bottom=0.0)
        ax2.set_ylabel("Probability Density")

        ax1.set_title(title)
        ax1.set_xscale("log")
        ax1.set_ylabel("Count (Frequency)")
        ax1.set_xlabel("Value (Log Scale)")

        # 合并图例
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        fig.tight_layout()
        return fig, (sigma, scale)
