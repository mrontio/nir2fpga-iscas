from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import setVCD

if TYPE_CHECKING:
    from InternalSimulator.NIR2FPGA import NIR2FPGA
    from InternalSimulator.HardwareVariables import HardwareVariables


@dataclass
class SimulationOptions:
    """Options bag for hardware simulation — mirrors DiscretizationChoices."""
    output_check: bool = False
    output_check_precision: Optional[int] = None  # None = inherit Scala default (output frac bits)
    reduction: Optional[bool] = None   # None = inherit from dc.reduction
    mac_width: Optional[int] = None    # None = inherit from dc.macWidth
    spike_gating: bool = True
    ignore_timestamp: bool = False
    primitives_dir: Optional[str] = None  # path relative to 2-compilation/, passed as -DprimitivesDir


class HardwareSimulation:
    """
    Responsible for running and caching hardware simulation output.

    Initialised in NIR2FPGA.__init__. Holds simulation options,
    the timestamp of the last successful sbt run, and handles
    VCD discovery and caching.
    """

    def __init__(self, n2f: "NIR2FPGA", options: SimulationOptions) -> None:
        self.n2f = n2f
        self.options = options
        self.last_run_time: Optional[float] = None
        self._compile_hash: Optional[str] = None
        self._compile_successful: bool = False
        self._compilation_dir: Optional[Path] = None
        self._vcd_path: Optional[Path] = None
        self._setvcd: Optional[Any] = None
        self._vcd_hash: str = ""
        self.layer_to_neuron_map: Dict[str, str] = {}
        self.layer_to_linear_map: Dict[str, str] = {}
        self._hw_vars: Optional["HardwareVariables"] = None

    # ── VCD / path discovery ───────────────────────────────────────────────

    def _find_compilation_dir(self) -> Path:
        """Walk upward from this file to find the 2-compilation/ directory."""
        candidate = Path(__file__).resolve()
        for _ in range(10):
            candidate = candidate.parent
            target = candidate / "2-compilation"
            if target.is_dir():
                return target
        raise FileNotFoundError(
            f"Could not find '2-compilation/' by walking up from {Path(__file__).resolve()}"
        )

    @property
    def compilation_dir(self) -> Path:
        if self._compilation_dir is None:
            self._compilation_dir = self._find_compilation_dir()
        return self._compilation_dir

    @property
    def effective_primitives_dir(self) -> Path:
        """Resolve the primitives directory: custom if set, otherwise the default."""
        if self.options.primitives_dir is not None:
            p = Path(self.options.primitives_dir)
            return p if p.is_absolute() else self.compilation_dir / p
        return self.compilation_dir / "hw/spinal/NIR2FPGA/src/primitives"

    def _find_vcd(self) -> Optional[Path]:
        """
        Search simWorkspace for wave.fst (preferred) or wave.vcd files containing
        the configTimestamp signal. FST files are binary so setVCD is used to check;
        VCD files are scanned as text up to $enddefinitions.
        """
        sim_workspace = self.compilation_dir / "simWorkspace"
        if not sim_workspace.exists():
            return None

        # Prefer FST (produced by withFstWave); fall back to VCD.
        for pattern, is_fst in [("**/wave.fst", True), ("**/wave.vcd", False)]:
            for wave_file in sorted(sim_workspace.glob(pattern)):
                try:
                    if is_fst:
                        svcd = setVCD.SetVCD(str(wave_file), "AcceleratorAXI.clk")
                        if svcd.search(r"configTimestamp"):
                            return wave_file
                    else:
                        with open(wave_file, "r", errors="replace") as fh:
                            for line in fh:
                                if "configTimestamp" in line:
                                    return wave_file
                                if "$enddefinitions" in line:
                                    break
                except OSError:
                    continue
        return None

    @property
    def vcd_path(self) -> Path:
        """Resolve (and cache) the VCD path, raising if not found."""
        if self._vcd_path is None:
            found = self._find_vcd()
            if found is None:
                raise FileNotFoundError(
                    "[HardwareSimulation] No FST/VCD file containing 'configTimestamp' found "
                    f"under {self.compilation_dir / 'simWorkspace'}. "
                    "Run the hardware simulation first (call simulate() or run())."
                )
            self._vcd_path = found
        return self._vcd_path

    def _hash_primitives_dir(self) -> str:
        """SHA-256 over all file contents in the effective primitives directory, sorted by path."""
        pdir = self.effective_primitives_dir
        if not pdir.is_dir():
            raise FileNotFoundError(f"[HardwareSimulation] Primitives dir not found: {pdir}")
        h = hashlib.sha256()
        for path in sorted(pdir.rglob("*")):
            if path.is_file():
                h.update(path.read_bytes())
        return h.hexdigest()

    # ── Simulation runner ──────────────────────────────────────────────────

    def run(
        self,
        inputs_path: Path,
        recordings_path: Optional[Path] = None,
        precision: Optional[int] = None,
    ) -> None:
        """
        Execute sbt simulation. Sets last_run_time on success and
        invalidates the VCD cache so get_vcd() re-reads the fresh file.

        precision: fractional-bit resolution for recordings comparison
        (None = inherit options.output_check_precision, else Scala default).
        """
        use_precision = precision if precision is not None else self.options.output_check_precision
        reduction = self.options.reduction if self.options.reduction is not None else self.n2f.dc.reduction
        mac_width = self.options.mac_width if self.options.mac_width is not None else self.n2f.dc.macWidth

        files_dir = self.n2f.files_dir
        if files_dir is None:
            files_dir = self.compilation_dir / "inputs" / self.n2f.filename
        model_json = files_dir / "model.json"
        if not model_json.exists():
            raise FileNotFoundError(
                f"[HardwareSimulation] Model files not found at {files_dir}. "
                "Call save_files() first."
            )

        flags = f" --input_packets={inputs_path}"
        if recordings_path is not None:
            flags += f" --recordings_path={recordings_path}"

        if use_precision is not None:
            flags += f" --precision={use_precision}"
        if reduction:
            flags += " --reduction=true"
        if not self.options.spike_gating:
            flags += " --spikeGating=false"
        if mac_width != 4:
            flags += f" --macWidth={mac_width}"

        jvm_props = ""
        if self.options.primitives_dir is not None:
            jvm_props = f"-DprimitivesDir={self.options.primitives_dir} "
        cmd = f'sbt {jvm_props}"runMain NIR2FPGA.Test {files_dir} {flags}"'
        print(f"[HardwareSimulation] Running: {cmd}")

        result = subprocess.run(cmd, shell=True, cwd=self.compilation_dir)
        if result.returncode != 0:
            raise RuntimeError(
                f"[HardwareSimulation] Simulation failed (exit code {result.returncode})"
            )

        self.last_run_time = time.time()
        self._setvcd = None
        self._vcd_hash = ""
        self._vcd_path = None
        print("[HardwareSimulation] Simulation complete.")

    def compile(self, output_dir: Optional[Path] = None) -> None:
        """
        Run sbt Generate to produce AcceleratorAXI Verilog.

        output_dir: directory under which Verilog is written (default: <compilation_dir>/outputs).
        Records success and hashes the primitives directory so callers can detect stale
        primitives without re-running a full simulation.
        """
        self._compile_successful = False

        files_dir = self.n2f.files_dir
        if files_dir is None:
            files_dir = self.compilation_dir / "inputs" / self.n2f.filename
        model_json = files_dir / "model.json"
        if not model_json.exists():
            raise FileNotFoundError(
                f"[HardwareSimulation] Model files not found at {files_dir}. "
                "Call save_files() first."
            )

        if output_dir is None:
            output_dir = self.compilation_dir / "outputs"

        jvm_props = ""
        if self.options.primitives_dir is not None:
            jvm_props = f"-DprimitivesDir={self.options.primitives_dir} "
        cmd = f'sbt {jvm_props}"runMain NIR2FPGA.Generate {files_dir} {output_dir}"'
        print(f"[HardwareSimulation] Generating Verilog: {cmd}")

        result = subprocess.run(cmd, shell=True, cwd=self.compilation_dir)
        if result.returncode != 0:
            raise RuntimeError(
                f"[HardwareSimulation] Verilog generation failed (exit code {result.returncode})"
            )

        self._compile_successful = True
        self._compile_hash = self._hash_primitives_dir()
        print(f"[HardwareSimulation] Verilog generation successful. Primitives hash: {self._compile_hash[:12]}…")

    # ── VCD accessor ───────────────────────────────────────────────────────

    def get_vcd(
        self,
        ignore_timestamp: Optional[bool] = None,
        vcd_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Return a cached setVCD.SetVCD. Runs simulation if VCD is missing
        or stale relative to model.json. Validates configTimestamp unless
        ignore_timestamp is True (defaults to options.ignore_timestamp).
        """
        actual_ignore = ignore_timestamp if ignore_timestamp is not None else self.options.ignore_timestamp

        # Resolve the path: explicit override takes precedence over tree search.
        if vcd_path is not None:
            resolved = Path(vcd_path).absolute()
        else:
            resolved = self._find_vcd()

        files_dir = self.n2f.files_dir or (self.compilation_dir / "inputs" / self.n2f.filename)
        model_json = files_dir / "model.json"

        needs_run = False
        if resolved is None or not resolved.exists():
            print("[HardwareSimulation] VCD not found, running simulation...")
            needs_run = True
        elif model_json.exists() and resolved.stat().st_mtime < model_json.stat().st_mtime:
            if actual_ignore:
                print("[HardwareSimulation] VCD is stale, but ignore_timestamp=True — skipping re-run.")
            else:
                print("[HardwareSimulation] VCD is stale, re-running simulation...")
                needs_run = True

        if needs_run:
            inputs_path = files_dir / "input_packets.npy"
            self.run(inputs_path=inputs_path)
            resolved = self._find_vcd() if vcd_path is None else resolved

        assert resolved is not None

        vcd_hash = hashlib.sha256(open(resolved, "rb").read()).hexdigest()
        if self._setvcd is not None and vcd_hash == self._vcd_hash:
            return self._setvcd  # cached hit

        file_size = os.path.getsize(resolved)
        if file_size > 10 * 1024 * 1024:
            print(
                f"[HardwareSimulation] Loading VCD "
                f"({file_size / (1024 * 1024):.1f} MB) — may take a while..."
            )

        self._vcd_hash = vcd_hash
        self._vcd_path = resolved
        setvcd = setVCD.SetVCD(str(resolved), "AcceleratorAXI.clk")
        self._setvcd = setvcd
        print("[HardwareSimulation] VCD loaded.")

        if not actual_ignore:
            vcd_timestamp_signal = r"configTimestamp"
            timestamp_signals = setvcd.search(vcd_timestamp_signal)
            if not timestamp_signals:
                raise KeyError(
                    f"[HardwareSimulation] Timestamp signal '{vcd_timestamp_signal}' "
                    f"not found in {resolved}"
                )
            clock_sigs = setvcd.search("AcceleratorAXI.clk")
            rising = setvcd.get(clock_sigs[0], lambda x, y: x == 0 and y == 1)
            ts_values = setvcd.get_values(timestamp_signals[0], rising)
            if not ts_values:
                raise KeyError(
                    f"[HardwareSimulation] No values found for timestamp signal '{vcd_timestamp_signal}'"
                )
            timestamp = ts_values[0]
            if timestamp != self.n2f.timestamp:
                theirs = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                ours = datetime.fromtimestamp(self.n2f.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                raise Exception(
                    f"[HardwareSimulation] VCD timestamp ({theirs}) does not match "
                    f"model timestamp ({ours})"
                )

        return self._setvcd

    # ── HardwareVariables integration ──────────────────────────────────────────

    def simulate(self,
                 sample: Any,
                 output_recording: Optional[Any] = None,
                 skip_vcd = False
                 ) -> None:
        """Run simulation and build HardwareVariables for VCD signal access.

        If sample is provided (shape (T, N)), generates packets from it and
        runs sbt with those packets so hardware recordings correspond to that sample.
        """
        from InternalSimulator.HardwareVariables import HardwareVariables

        try:
            import torch as _torch
            sample_np = sample.detach().numpy() if isinstance(sample, _torch.Tensor) else np.array(sample)
        except ImportError:
            sample_np = np.array(sample)
        packets = self.n2f.io_manager.create_input_packets_numpy(sample_np)
        files_dir = self.n2f.files_dir or (self.compilation_dir / "inputs" / self.n2f.filename)
        inputs_path = files_dir / "input_packets.npy"
        np.save(str(inputs_path), packets)

        recordings_path = None
        if output_recording is not None:
            try:
                import torch as _torch
                recording_np = output_recording.detach().numpy() if isinstance(output_recording, _torch.Tensor) else np.array(output_recording)
            except ImportError:
                recording_np = np.array(output_recording)
            files_dir = self.n2f.files_dir or (self.compilation_dir / "inputs" / self.n2f.filename)
            recordings_path = files_dir / "recordings.npy"
            np.save(str(recordings_path), recording_np)


        self.run(inputs_path=inputs_path, recordings_path=recordings_path)

        if not skip_vcd:
            setvcd = self.get_vcd(ignore_timestamp=self.options.ignore_timestamp)
            self._hw_vars = HardwareVariables(
                setvcd=setvcd,
                nir_graph=self.n2f.nir_graph,
                internal_model=self.n2f.internal_model,
                quantization_data=self.n2f.quantization_data,
                timesteps=self.n2f.timesteps,
            )

    def _require_hw_vars(self) -> "HardwareVariables":
        if self._hw_vars is None:
            raise RuntimeError(
                "[HardwareSimulation] Hardware variables not initialised. "
                "Call simulate() first."
            )
        return self._hw_vars

    def get_nodes(self) -> List[str]:
        """Layer IDs for all hardware-observable layers."""
        return self._require_hw_vars().get_nodes()

    def get_parameters(self, node: str) -> List[str]:
        """Recordable parameter names for a given layer ID."""
        return self._require_hw_vars().get_parameters(node)

    def get_recording(
        self,
        node: str,
        index: int,
        parameters: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return a dict of parameter → (timesteps,) numpy array for the given node and index."""
        return self._require_hw_vars().get_recording(node, index, parameters)

    # ── Accelerator output loader ───────────────────────────────────────────────

    def load_accelerator_output(
        self,
        ignore_timestamp: bool = False,
        vcd_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Decode the AXI output stream from the VCD and return the output tensor."""
        import InternalSimulator.VCDMapping as VCDMapping
        import time as _time

        start = _time.time()
        vcd = self.get_vcd(ignore_timestamp=ignore_timestamp, vcd_path=vcd_path)
        sigs = VCDMapping.resolve_global_signals(vcd)
        print(f"Loaded VCD in {_time.time() - start:.2f}s")

        one = lambda x: x == 1
        rising_edges = vcd.get(sigs["clock"], lambda x, y: x == 0 and y == 1)
        reset0 = vcd.get(sigs["reset"], lambda x: x == 0)
        valid = vcd.get(sigs["m_axis_valid"], one)
        ready = vcd.get(sigs["m_axis_ready"], one)
        handshake_fire = rising_edges & reset0 & valid & ready
        print(f"Defined signal condition in {_time.time() - start:.2f}s")

        output_packets = vcd.get_values(sigs["m_axis_data"], handshake_fire)
        print(f"Applied signal condition in {_time.time() - start:.2f}s")
        return self.n2f.io_manager.decode_output_packets(output_packets)
