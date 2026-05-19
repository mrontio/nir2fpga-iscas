from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass
class PTQOptions:
    dataset: Optional[Dataset] = None
    batch_size: Optional[int] = None
    num_samples: Optional[int] = None
    seed: int = 42
    method: str = "minmax"
    percentile: Optional[float] = None
    threshold_headroom_multiplier: float = 2.0
    readout_percentile: Optional[float] = None
    histogram_bins: int = 256
    histogram_sample_stride: int = 1


@dataclass
class DiscretizationChoices:
    timesteps: int
    dataset: Dataset
    transform: Optional[T.Compose] = None
    batch_size: int = 64
    total_bits: int = 16
    weight_bits: Optional[int] = None
    dt_ms: float = 1
    reduction: bool = False
    macWidth: int = 4
    dataset_name: str = "skip"
    ptq: Optional[PTQOptions] = None

    # Legacy calibration fields retained for backwards compatibility.
    calibration_dataset: Optional[Dataset] = None
    calibration_batch_size: Optional[int] = None
    calibration_num_samples: Optional[int] = None
    calibration_seed: int = 42
    calibration_method: str = "minmax"
    calibration_percentile: Optional[float] = None
    threshold_headroom_multiplier: float = 2.0
    readout_percentile: Optional[float] = None
    histogram_bins: int = 256
    histogram_sample_stride: int = 1
    representative_sample_index: int = 0
    benchmark_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.ptq is None:
            self.ptq = PTQOptions(
                dataset=self.calibration_dataset,
                batch_size=self.calibration_batch_size,
                num_samples=self.calibration_num_samples,
                seed=self.calibration_seed,
                method=self.calibration_method,
                percentile=self.calibration_percentile,
                threshold_headroom_multiplier=self.threshold_headroom_multiplier,
                readout_percentile=self.readout_percentile,
                histogram_bins=self.histogram_bins,
                histogram_sample_stride=self.histogram_sample_stride,
            )
        else:
            self.calibration_dataset = self.ptq.dataset
            self.calibration_batch_size = self.ptq.batch_size
            self.calibration_num_samples = self.ptq.num_samples
            self.calibration_seed = self.ptq.seed
            self.calibration_method = self.ptq.method
            self.calibration_percentile = self.ptq.percentile
            self.threshold_headroom_multiplier = self.ptq.threshold_headroom_multiplier
            self.readout_percentile = self.ptq.readout_percentile
            self.histogram_bins = self.ptq.histogram_bins
            self.histogram_sample_stride = self.ptq.histogram_sample_stride

    def effective_calibration_dataset(self) -> Dataset:
        assert self.ptq is not None
        return self.ptq.dataset if self.ptq.dataset is not None else self.dataset

    def effective_calibration_batch_size(self) -> int:
        assert self.ptq is not None
        return self.ptq.batch_size if self.ptq.batch_size is not None else self.batch_size

    def metadata(self) -> dict[str, Any]:
        assert self.ptq is not None
        return {
            "benchmark_name": self.benchmark_name,
            "timesteps": self.timesteps,
            "batch_size": self.batch_size,
            "total_bits": self.total_bits,
            "weight_bits": self.weight_bits,
            "dt_ms": self.dt_ms,
            "reduction": self.reduction,
            "macWidth": self.macWidth,
            "dataset_name": self.dataset_name,
            "calibration_batch_size": self.effective_calibration_batch_size(),
            "calibration_num_samples": self.ptq.num_samples,
            "calibration_seed": self.ptq.seed,
            "calibration_method": self.ptq.method,
            "calibration_percentile": self.ptq.percentile,
            "threshold_headroom_multiplier": self.ptq.threshold_headroom_multiplier,
            "readout_percentile": self.ptq.readout_percentile,
            "histogram_bins": self.ptq.histogram_bins,
            "histogram_sample_stride": self.ptq.histogram_sample_stride,
            "representative_sample_index": self.representative_sample_index,
        }
