import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nir
import numpy as np
from typing import Dict, Optional


class Plotter:
    """
    Visualizer for hardware and simulation recordings.

    Constructed once per NIR2FPGA instance. Dispatches to node-type-specific
    plot methods using the normalised NIR graph. All recording arrays are
    expected to be 1D (T,) — neuron slicing happens before this class.
    """

    def __init__(self, nir_graph: nir.NIRGraph) -> None:
        self._nir_graph = nir_graph
        self.styles: Dict[str, Dict[str, object]] = {
            "source": {
                "color": "#d62728",
                "linestyle": ":",
                "alpha": 0.6,
                "linewd": 1,
                "pointwd": 50,
                "marker": "v",
            },
            "internal": {
                "color": "#1f77b4",
                "linestyle": "-",
                "alpha": 0.9,
                "linewd": 2.5,
                "pointwd": 200,
                "marker": "s",
            },
            "quantized": {
                "color": "#ff7f0e",
                "linestyle": "--",
                "alpha": 0.8,
                "linewd": 2.0,
                "pointwd": 150,
                "marker": "o",
            },
            "behavioural": {
                "color": "#9467bd",
                "linestyle": "-.",
                "alpha": 0.75,
                "linewd": 1.8,
                "pointwd": 120,
                "marker": "D",
            },
            "hardware": {
                "color": "#2ca02c",
                "linestyle": "-",
                "alpha": 0.7,
                "linewd": 1.5,
                "pointwd": 100,
                "marker": "^",
            },
        }

    def _validate_recording(
        self,
        name: str,
        recording: Dict[str, np.ndarray],  # type: ignore[type-arg]
        required_keys: Optional[set] = None,  # type: ignore[type-arg]
    ) -> None:
        if required_keys is None:
            required_keys = {"output"}
        for key in required_keys & set(recording.keys()):
            arr = recording[key]
            if not isinstance(arr, np.ndarray):
                raise ValueError(
                    f"Recording '{name}[{key}]' must be a numpy array, "
                    f"got {type(arr).__name__}"
                )
            if arr.ndim != 1:
                raise ValueError(
                    f"Recording '{name}[{key}]' must be 1D (T,), got shape {arr.shape}"
                )

    def _collect_recordings(
        self,
        recordings: Dict[str, Dict[str, np.ndarray]],  # type: ignore[type-arg]
        required_keys: set,  # type: ignore[type-arg]
    ) -> Dict[str, Dict[str, np.ndarray]]:  # type: ignore[type-arg]
        """Validate all recordings and return only those that are non-empty."""
        result: Dict[str, Dict[str, np.ndarray]] = {}  # type: ignore[type-arg]
        for name, rec in recordings.items():
            self._validate_recording(name, rec, required_keys=required_keys & set(rec.keys()))
            result[name] = rec
        if not result:
            raise ValueError("At least one recording must be provided")
        return result

    def _finish_plot(self, filename: Optional[str]) -> None:
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def _plot_lif(
        self,
        recordings: Dict[str, Dict[str, np.ndarray]],  # type: ignore[type-arg]
        filename: Optional[str],
        plot_size: Optional[tuple[int, int]],
    ) -> None:
        """2-subplot: v_mem (line) + spikes (scatter). For LIF, IF, CubaLIF nodes."""
        recs = self._collect_recordings(recordings, required_keys={"output"})
        figsize = plot_size if plot_size is not None else (14, 9)
        _, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        for plot_i, (key, recording) in enumerate(recs.items()):
            style = self.styles[key]
            if "v_mem" in recording:
                v_mem = recording["v_mem"]
                ax1.plot(
                    range(len(v_mem)),
                    v_mem,
                    label=f"{key.capitalize()} v_mem",
                    linewidth=style["linewd"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    marker=style["marker"],
                    markersize=4,
                )
            if "output" in recording:
                spikes = recording["output"]
                timesteps = np.arange(len(spikes))
                nonzero = spikes > 0
                if nonzero.any():
                    ax2.scatter(
                        timesteps[nonzero],
                        spikes[nonzero],
                        s=style["pointwd"],
                        marker=style["marker"],
                        color=style["color"],
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=0.8,
                        zorder=5 + plot_i,
                        label=f"{key.capitalize()} Spikes",
                    )

        ax1.set_ylabel("v_mem", fontsize=12)
        ax1.set_title("v_mem and spikes", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax2.set_xlabel("Timestep", fontsize=12)
        ax2.set_ylabel("Spike Count", fontsize=12)
        ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.set_ylim(0, 2)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        self._finish_plot(filename)

    def _plot_li(
        self,
        recordings: Dict[str, Dict[str, np.ndarray]],  # type: ignore[type-arg]
        filename: Optional[str],
        plot_size: Optional[tuple[int, int]],
    ) -> None:
        """1-subplot: v_mem only. For LI nodes."""
        recs = self._collect_recordings(recordings, required_keys={"output"})
        figsize = plot_size if plot_size is not None else (14, 5)
        _, ax = plt.subplots(1, 1, figsize=figsize)

        for key, recording in recs.items():
            style = self.styles[key]
            for signal in ("output", "v_mem"):
                if signal not in recording:
                    continue
                arr = recording[signal]
                ax.plot(
                    range(len(arr)),
                    arr,
                    label=f"{key.capitalize()} {signal}",
                    linewidth=style["linewd"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    marker=style["marker"],
                    markersize=4,
                )

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("value", fontsize=12)
        ax.set_title("output", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        self._finish_plot(filename)

    def _plot_affine(
        self,
        recordings: Dict[str, Dict[str, np.ndarray]],  # type: ignore[type-arg]
        filename: Optional[str],
        plot_size: Optional[tuple[int, int]],
    ) -> None:
        """1-subplot: output only. For Affine/Linear nodes."""
        recs = self._collect_recordings(recordings, required_keys={"output"})
        figsize = plot_size if plot_size is not None else (14, 5)
        _, ax = plt.subplots(1, 1, figsize=figsize)

        for key, recording in recs.items():
            style = self.styles[key]
            if "output" in recording:
                output = recording["output"]
                ax.plot(
                    range(len(output)),
                    output,
                    label=f"{key.capitalize()} output",
                    linewidth=style["linewd"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    marker=style["marker"],
                    markersize=4,
                )

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Output", fontsize=12)
        ax.set_title("Output", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        self._finish_plot(filename)

    def plot(
        self,
        node_id: str,
        recordings: Dict[str, Dict[str, np.ndarray]],  # type: ignore[type-arg]
        filename: Optional[str] = None,
        plot_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Plot recordings for node_id, dispatching on its NIR type.

        recordings: dict mapping rec_type (e.g. "internal", "quantized") to
                    {param_name: 1D numpy array of shape (T,)}.

        Raises KeyError for unknown node_id.
        Raises ValueError for unsupported NIR types.
        """
        if node_id not in self._nir_graph.nodes:
            raise KeyError(f"Node '{node_id}' not found in NIR graph.")
        node = self._nir_graph.nodes[node_id]

        if isinstance(node, (nir.LIF, nir.IF, nir.CubaLIF)):
            self._plot_lif(recordings, filename, plot_size)
        elif isinstance(node, (nir.LI, nir.I)):
            self._plot_li(recordings, filename, plot_size)
        elif isinstance(node, (nir.Affine, nir.Linear)):
            self._plot_affine(recordings, filename, plot_size)
        else:
            raise ValueError(
                f"Plotting NIR type {type(node).__name__} is not yet supported."
            )
