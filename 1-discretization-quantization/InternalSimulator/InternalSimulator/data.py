from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from InternalSimulator.DiscretizationChoices import DiscretizationChoices


class DeterministicRateEncoder:
    """Deterministic rate encoder for image-like inputs.

    Each sample index gets its own RNG seed, so repeated reads of the same item
    always produce the same spike train regardless of DataLoader order.
    """

    def __init__(self, timesteps: int = 100, gain: float = 1.0, base_seed: int = 42):
        self.timesteps = timesteps
        self.gain = gain
        self.base_seed = base_seed

    def __call__(self, img: torch.Tensor, index: int) -> torch.Tensor:
        flat = img.reshape(-1).float()
        probs = torch.clamp(flat * self.gain, min=0.0, max=1.0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.base_seed + int(index))
        samples = torch.rand((self.timesteps, probs.numel()), generator=generator)
        return (samples < probs.unsqueeze(0)).float()


class DeterministicRateEncodedMNIST(Dataset):
    """MNIST wrapper that produces deterministic rate-coded spike trains."""

    def __init__(
            self,
            train: bool,
            timesteps: int = 100,
            gain: float = 1.0,
            data_dir: Optional[Union[str, Path]] = None,
            seed: int = 42,
    ) -> None:
        from torchvision import datasets, transforms

        if data_dir is None:
            data_dir = Path.home() / ".cache/nir-fpga/data"

        base_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])

        self.base_dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=base_transform,
        )
        self.encoder = DeterministicRateEncoder(
            timesteps=timesteps,
            gain=gain,
            base_seed=seed + (0 if train else 1_000_000),
        )
        self.timesteps = timesteps

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img, label = self.base_dataset[index]
        encoded = self.encoder(img, index)
        return encoded, int(label)


def random_spikes(
        num_samples: int = 100,
        num_timesteps: int = 1000,
        num_inputs: int = 1,
        min_spikes: int = 10,
        max_spikes: int = 100,
        spike_val: float = 1.0,
        seed: int = 42,
        batch_size: Optional[int] = None,
) -> DiscretizationChoices:
    """Random spike data. Shape per sample: (timesteps, num_inputs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = torch.zeros((num_samples, num_timesteps, num_inputs))
    for sample_idx in range(num_samples):
        num_spikes = np.random.randint(min_spikes, max_spikes + 1)
        spike_positions = np.random.choice(num_timesteps, size=num_spikes, replace=False)
        dataset[sample_idx, spike_positions, 0] = spike_val

    labels = torch.zeros(num_samples, dtype=torch.long)
    bs = batch_size if batch_size is not None else num_samples
    return DiscretizationChoices(
        timesteps=num_timesteps,
        dataset=TensorDataset(dataset, labels),
        batch_size=bs,
    )


def nir_paper_spikes(timesteps: int = 1000, batch_size: Optional[int] = None) -> DiscretizationChoices:
    """Hardcoded 100-step spike pattern from NIR paper, expanded to timesteps."""
    d100 = [
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    expansion = timesteps // 100
    expanded = []
    for t in d100:
        expanded.append(t)
        for _ in range(expansion - 1):
            expanded.append(0)

    data = torch.tensor(expanded).unsqueeze(1).unsqueeze(0).float()
    labels = torch.zeros(1, dtype=torch.long)
    bs = batch_size if batch_size is not None else 1
    return DiscretizationChoices(
        timesteps=timesteps,
        dataset=TensorDataset(data, labels),
        batch_size=bs,
    )


def random_linear_spikes(
        num_samples: int = 100,
        num_timesteps: int = 100,
        num_inputs: int = 10,
        min_spikes: Optional[int] = None,
        max_spikes: Optional[int] = None,
        seed: int = 42,
        batch_size: Optional[int] = None,
) -> DiscretizationChoices:
    """Random spikes for multi-input linear layers."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    spikes_min: int = min_spikes if min_spikes else 1
    spikes_max: int = max_spikes if max_spikes else int(0.5 * num_timesteps)

    dataset = torch.zeros((num_samples, num_timesteps, num_inputs))
    for sample_idx in range(num_samples):
        num_spikes = np.random.randint(spikes_min, spikes_max + 1)
        spike_positions = np.random.choice(num_timesteps, size=num_spikes, replace=False)
        dataset[sample_idx, spike_positions, 0] = 1.0

    labels = torch.zeros(num_samples, dtype=torch.long)
    bs = batch_size if batch_size is not None else num_samples
    return DiscretizationChoices(
        timesteps=num_timesteps,
        dataset=TensorDataset(dataset, labels),
        batch_size=bs,
    )


def nmnist(
        timesteps: int = 100,
        batch_size: int = 64,
        train: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
) -> DiscretizationChoices:
    """N-MNIST via tonic."""
    import tonic
    import tonic.datasets as tonic_datasets
    import tonic.transforms as tonic_tf

    class FlattenSensor:
        def __call__(self, data: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
            return data.reshape(data.shape[0], -1)

    sensor_size = tonic_datasets.NMNIST.sensor_size
    snn_transform = tonic.transforms.Compose([  # type: ignore[arg-type]
        tonic_tf.ToFrame(sensor_size=sensor_size, n_time_bins=timesteps),
        FlattenSensor(),
    ])

    if cache_dir is None:
        cache_dir = Path.home() / "data"
        cache_dir = Path(cache_dir)

    dataset = tonic.datasets.NMNIST(train=train, transform=snn_transform, save_to=str(cache_dir))
    cache_subdir = "train" if train else "test"
    dataset = tonic.DiskCachedDataset(dataset, cache_path=str(cache_dir / cache_subdir))  # type: ignore[arg-type]

    return DiscretizationChoices(
        timesteps=timesteps,
        dataset=dataset,  # type: ignore[arg-type]
        transform=snn_transform,  # type: ignore[arg-type]
        batch_size=batch_size,
    )


def shd(
        timesteps: int = 100,
        batch_size: int = 64,
        train: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
) -> DiscretizationChoices:
    """SHD (Spiking Heidelberg Digits) via tonic."""
    import tonic
    import tonic.datasets as tonic_datasets
    import tonic.transforms as tonic_tf

    class FlattenSensor:
        def __call__(self, data: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
            return data.reshape(data.shape[0], -1)

    sensor_size = tonic_datasets.SHD.sensor_size
    snn_transform = tonic.transforms.Compose([  # type: ignore[arg-type]
        tonic_tf.ToFrame(sensor_size=sensor_size, n_time_bins=timesteps),
        FlattenSensor(),
    ])

    if cache_dir is None:
        cache_dir = Path.home() / "data"
        cache_dir = Path(cache_dir)

    dataset = tonic.datasets.SHD(train=train, transform=snn_transform, save_to=str(cache_dir))
    cache_subdir = "shd_train" if train else "shd_test"
    dataset = tonic.DiskCachedDataset(dataset, cache_path=str(cache_dir / cache_subdir))  # type: ignore[arg-type]

    return DiscretizationChoices(
        timesteps=timesteps,
        dataset=dataset,  # type: ignore[arg-type]
        transform=snn_transform,  # type: ignore[arg-type]
        batch_size=batch_size,
    )


def mnist(
        batch_size: int = 64,
        train: bool = False,
        data_dir: Optional[Union[str, Path]] = None,
        timesteps: int = 100,
        gain: float = 1.0,
        deterministic: bool = True,
        seed: int = 42,
) -> DiscretizationChoices:
    """MNIST with rate-coded spike encoding.

    When ``deterministic=True`` (default), each dataset index maps to a stable
    spike train, making quantization benchmarks reproducible.
    """
    from snntorch import spikegen
    from torchvision import datasets, transforms

    if data_dir is None:
        data_dir = Path.home() / ".cache/nir-fpga/data"

    if deterministic:
        dataset = DeterministicRateEncodedMNIST(
            train=train,
            timesteps=timesteps,
            gain=gain,
            data_dir=data_dir,
            seed=seed,
        )
        transform = None
    else:
        class SpikeTransform:
            def __init__(self, timesteps: int = 100, gain: float = 1.0):
                self.timesteps = timesteps
                self.gain = gain

            def __call__(self, img: torch.Tensor) -> torch.Tensor:
                flat = img.reshape(img.shape[1] * img.shape[2])
                return spikegen.rate(flat, num_steps=self.timesteps, gain=self.gain)  # pyright: ignore[reportArgumentType]

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            SpikeTransform(timesteps=timesteps, gain=gain),
        ])

        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )

    return DiscretizationChoices(
        timesteps=timesteps,
        dataset=dataset,
        transform=transform,
        batch_size=batch_size,
    )


def from_tensor(
        data: Any,
        labels: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
) -> DiscretizationChoices:
    """Wrap an existing tensor as a DiscretizationChoices."""
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()
    if labels is None:
        labels = torch.zeros(data.shape[0], dtype=torch.long)
    bs = batch_size if batch_size is not None else data.shape[0]
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    return DiscretizationChoices(
        timesteps=data.shape[1],
        dataset=TensorDataset(data, labels),
        batch_size=bs,
    )


# Dataset registry mapping names to default paths and loader factory functions
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "mnist": {
        "path": Path.home() / "data" / "mnist_test_frames",
        "loader_factory": mnist,
    },
    "shd": {
        "path": Path.home() / "data" / "shd_test_frames",
        "loader_factory": shd,
    },
    "nmnist": {
        "path": Path.home() / "data" / "nmnist",
        "loader_factory": None,  # N-MNIST uses tonic loaders, not pre-saved frames
    },
}


def get_dataset_path(dataset_name: str) -> Path:
    """Get the default dataset root path by name.

    Args:
        dataset_name: Dataset identifier ("mnist", "shd", "nmnist", etc.)

    Returns:
        Path object pointing to dataset root

    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]["path"]


def get_sample_by_index(dataset_name: str, index: int, root: Optional[Path] = None) -> tuple:
    """Load a single sample by index from a pre-saved dataset.

    Samples are indexed lexicographically by filename ({idx:05d}_{label}.npy).
    This function works for datasets that store samples as individual .npy files
    (mnist, shd). For streaming datasets like nmnist, use the appropriate loader function.

    Args:
        dataset_name: Dataset identifier ("mnist", "shd", etc.)
        index: Zero-based sample index
        root: Optional override for dataset root path. If None, uses registry default.

    Returns:
        (frame, label) tuple where frame is np.ndarray and label is int

    Raises:
        ValueError: If dataset not found, index out of range, or invalid dataset_name
        FileNotFoundError: If samples not found in directory
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_root: Path
    if root is not None:
        dataset_root = root
    else:
        dataset_root = DATASET_REGISTRY[dataset_name]["path"]

    # List all .npy files and sort lexicographically
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    filenames = sorted([f for f in dataset_root.iterdir() if f.suffix == ".npy"])
    filenames = [f.name for f in filenames]  # Extract names only

    if index < 0 or index >= len(filenames):
        raise ValueError(
            f"Index {index} out of range for dataset '{dataset_name}' "
            f"(available samples: {len(filenames)})"
        )

    filename = filenames[index]
    label = int(filename.split("_")[1].split(".")[0])
    frame_path = dataset_root / filename
    frame = np.load(str(frame_path))

    return frame, label
