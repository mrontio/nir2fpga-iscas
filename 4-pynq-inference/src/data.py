"""Standalone N-MNIST loader for PYNQ (no tonic dependency).

Parses AER binary files and yields (frame, label) tuples where
frame has shape (n_time_bins, 2312) and dtype int16.
"""

import os
import queue
import threading
from pathlib import Path
from typing import Iterator, Optional
import numpy as np

# Dataset registry mapping names to paths and loaders
# Update these paths to match your actual dataset storage locations
datasets_path = Path("/home") / "xilinx" / "jupyter_notebooks" / "nir2fpga" / "datasets"
DATASET_REGISTRY = {
    "mnist": {
        "path": datasets_path / "mnist_test_frames",
        "loader": None,  # Will be set to mnist_loader after function definition
    },
    "shd": {
        "path": datasets_path / "shd_test_frames",
        "loader": None,  # Will be set to shd_loader after function definition
    },
    "nmnist": {
        "path": datasets_path / "nmnist",
        "loader": None,  # Will be set to nmnist_loader after function definition
    },
}


def _read_bin(path) -> tuple:
    """Parse one N-MNIST .bin file into event arrays.

    Returns (x, y, t, p) numpy arrays for non-overflow events only.
    """
    with open(path, "rb") as fp:
        raw = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

    all_x = raw[0::5]
    all_y = raw[1::5]
    all_p = (raw[2::5] & 128) >> 7
    all_t = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]

    # Handle timestamp overflow events (y == 240 marks overflow)
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for idx in overflow_indices:
        all_t[idx:] += time_increment

    td = np.where(all_y != 240)[0]
    return all_x[td], all_y[td], all_t[td], all_p[td]


def _events_to_frame(x, y, t, p, n_time_bins: int) -> np.ndarray:
    """Bin events into a (n_time_bins, 2312) frame array (dtype int16).

    Replicates tonic's SliceByTimeBins exactly:
      time_window = (t_last - t_first) // n_time_bins
      Each bin i covers [t_first + i*stride, t_first + i*stride + time_window)
    Internally uses shape (n_time_bins, 2, 34, 34) before flattening.
    """
    frame = np.zeros((n_time_bins, 2, 34, 34), dtype=np.int16)

    if len(t) == 0:
        return frame.reshape(n_time_bins, 2312)

    # tonic sorts events by timestamp before slicing
    order = np.argsort(t, kind="stable")
    t_s, x_s, y_s, p_s = t[order], x[order], y[order], p[order]

    t_first, t_last = t_s[0], t_s[-1]
    time_window = (t_last - t_first) // n_time_bins  # integer division, overlap=0

    window_starts = np.arange(n_time_bins) * time_window + t_first
    window_ends = window_starts + time_window

    idx_start = np.searchsorted(t_s, window_starts, side="left")
    idx_end = np.searchsorted(t_s, window_ends, side="left")

    for i in range(n_time_bins):
        sl = slice(idx_start[i], idx_end[i])
        np.add.at(frame, (i, p_s[sl].astype(int), y_s[sl], x_s[sl]), 1)

    return frame.reshape(n_time_bins, 2312)


def nmnist_loader(root, n_time_bins: int = 100, train: bool = False, shuffle: bool = True) -> Iterator:
    """Yield (frame, label) pairs from an N-MNIST directory.

    Args:
        root: Path to dataset root containing Train/ and Test/ subdirs.
        n_time_bins: Number of time bins for the output frame.
        train: If True, load from Train/; otherwise from Test/.
        shuffle: If True (default), yield samples in random order.

    Yields:
        (frame, label): frame is np.ndarray shape (n_time_bins, 2312) dtype int16,
                        label is int in [0, 9].
    """
    import random

    root = Path(root)
    split_dir = root / ("Train" if train else "Test")

    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = int(class_dir.name)
        for bin_file in sorted(class_dir.glob("*.bin")):
            samples.append((bin_file, label))

    if shuffle:
        random.shuffle(samples)

    for bin_file, label in samples:
        x, y, t, p = _read_bin(bin_file)
        frame = _events_to_frame(x, y, t, p, n_time_bins)
        yield frame, label


def shd_loader(root, shuffle: bool = True) -> Iterator:
    """Yield (frame, label) pairs from pre-saved SHD .npy files.

    Args:
        root: Path to directory containing files named {idx:05d}_{label}.npy.
        shuffle: If True (default), yield samples in random order.

    Yields:
        (frame, label): frame is np.ndarray shape (100, 700) dtype int16,
                        label is int in [0, 19].
    """
    import random
    root_str = str(root)
    filenames = [f for f in os.listdir(root_str) if f.endswith(".npy")]
    filenames.sort()
    samples = []
    for f in filenames:
        label = int(f.split("_")[1].split(".")[0])
        samples.append((os.path.join(root_str, f), label))

    if shuffle:
        random.shuffle(samples)

    for path, label in samples:
        yield np.load(path), label


def mnist_loader(root, shuffle: bool = True) -> Iterator:
    """Yield (frame, label) pairs from pre-saved MNIST .npy files.

    Args:
        root: Path to directory containing files named {idx:05d}_{label}.npy.
        shuffle: If True (default), yield samples in random order.

    Yields:
        (frame, label): frame is np.ndarray shape (100, 784) dtype int16,
                        label is int in [0, 9].
    """
    import random
    root_str = str(root)
    # Use os.listdir + string sort instead of pathlib glob + Path sort
    # (pathlib __lt__ is extremely slow on ARM)
    filenames = [f for f in os.listdir(root_str) if f.endswith(".npy")]
    filenames.sort()
    samples = []
    for f in filenames:
        label = int(f.split("_")[1].split(".")[0])
        samples.append((os.path.join(root_str, f), label))

    if shuffle:
        random.shuffle(samples)

    for path, label in samples:
        yield np.load(path), label


# Finalize registry with loader functions
DATASET_REGISTRY["mnist"]["loader"] = mnist_loader
DATASET_REGISTRY["shd"]["loader"] = shd_loader
DATASET_REGISTRY["nmnist"]["loader"] = nmnist_loader


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
    """Load a single sample by index from the dataset.

    Samples are indexed lexicographically by filename ({idx:05d}_{label}.npy).

    Args:
        dataset_name: Dataset identifier ("mnist", "shd", "nmnist", etc.)
        index: Zero-based sample index
        root: Optional override for dataset root path. If None, uses registry default.

    Returns:
        (frame, label) tuple

    Raises:
        ValueError: If dataset not found, index out of range, or invalid dataset_name
        FileNotFoundError: If samples not found in directory
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )

    if root is None:
        root = DATASET_REGISTRY[dataset_name]["path"]
    else:
        root = Path(root)

    # List all .npy files and sort lexicographically
    root_str = str(root)
    if not os.path.isdir(root_str):
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    filenames = sorted([f for f in os.listdir(root_str) if f.endswith(".npy")])

    if index < 0 or index >= len(filenames):
        raise ValueError(
            f"Index {index} out of range for dataset '{dataset_name}' "
            f"(available samples: {len(filenames)})"
        )

    filename = filenames[index]
    label = int(filename.split("_")[1].split(".")[0])
    frame = np.load(os.path.join(root_str, filename))

    return frame, label


class PrefetchingLoader:
    """Loads and optionally pre-encodes samples in a background thread.

    Overlaps data I/O and packet encoding with FPGA inference.

    Args:
        loader: An iterator yielding (frame, label) tuples.
        io_manager: IOManager instance (optional). If provided, pre-encodes
                    frames into uint32 packet arrays. Yields (packets, label).
                    If None, yields raw (frame, label) tuples.
        prefetch_count: How many samples to buffer ahead (default 2).
    """

    def __init__(self, loader: Iterator, io_manager=None, prefetch_count: int = 2, include_raw: bool = False):
        self._loader = loader
        self._io_manager = io_manager
        self._include_raw = include_raw
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
        self._sentinel = object()
        self._error = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            for frame, label in self._loader:
                if self._io_manager is not None:
                    packets = self._io_manager.create_input_packets_numpy(frame)
                    if self._include_raw:
                        item = (frame, packets, label)
                    else:
                        item = (packets, label)
                else:
                    item = (frame, label)
                self._queue.put(item)
        except Exception as e:
            self._error = e
        self._queue.put(self._sentinel)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                if self._error is not None:
                    raise self._error
                return
            yield item
