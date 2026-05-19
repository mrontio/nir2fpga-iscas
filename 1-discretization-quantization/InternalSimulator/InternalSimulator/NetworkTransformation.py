import nir
import copy
import sinabs
import torch
import re
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, List, Callable, Tuple, cast
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset, Subset


from InternalSimulator.Quantization import QuantizationConfig
from InternalSimulator.CurrentFilter import CurrentFilterSqueeze
from InternalSimulator.Integrator import IntegratorSqueeze
from InternalSimulator.DiscretizationChoices import DiscretizationChoices
from InternalSimulator.Recording import Recording, RecordingType, RecordingCompleteness
from InternalSimulator.PredictionType import PredictionType, OutputEqual, LayerEqual

@dataclass
class ParameterSpace:
    pass

@dataclass
class Real(ParameterSpace):
    pass

@dataclass
class Quantized(ParameterSpace):
    properties: Dict[str, Any]

@dataclass
class NetworkStage:
    name: str
    parameter_space: ParameterSpace
    recordings: Recording
    model: Optional[torch.nn.Module] = None
    accuracy: Optional[float] = None

class NetworkTransformation:
    def __init__(self,
                 nir_graph: nir.NIRGraph,
                 dc: DiscretizationChoices,
                 source_recordings: Optional[Dict] = None):

        self._validate_normalized_nir_graph(nir_graph)
        self.nir_graph = nir_graph
        self.dc = dc
        first_data, _ = self.get_representative_sample()
        self.data_sample = first_data.unsqueeze(0)  # (1, T, N)
        self.labels = torch.cat([y for _, y in self.get_dataloader()], dim=0)
        self.calibration_dataset_size = len(cast(Any, self._materialize_calibration_dataset()))

        self.conversion_params: Dict = {
            "IAFSqueeze": {
                "spike_fn": sinabs.activation.spike_generation.SingleSpike,  # type: ignore[attr-defined]
                "reset_fn": sinabs.activation.MembraneReset(reset_value=0),  # type: ignore[attr-defined]
            },
            "LIFSqueeze": {
                "spike_fn": sinabs.activation.spike_generation.SingleSpike,  # type: ignore[attr-defined]
                "reset_fn": sinabs.activation.MembraneReset(reset_value=0),  # type: ignore[attr-defined]
                "norm_input": True,
            },
            "ExpLeakSqueeze": { # This is effectively LI
                "norm_input": False,
            },
            "Linear": {},
            "Identity": {},
        }

        self.stages: Dict[RecordingType, NetworkStage] = {}

        # INTERNAL stage (built first so _classify_recording can compare against it)
        internal_model = self._nir_to_internal(nir_graph)
        self.internal_ids = [str(i) for i in range(len(internal_model))]
        internal_recordings = self._execute_network(
            internal_model, self.data_sample, self.single_save_hook
        )
        internal_recording = Recording(RecordingType.INTERNAL, internal_recordings)
        self.stages[RecordingType.INTERNAL] = NetworkStage(
            name="Internal",
            parameter_space=Real(),
            recordings=internal_recording,
            model=internal_model,
        )

        # SOURCE stage (conditional, built after INTERNAL for classification)
        if source_recordings is not None:
            layer_types = {lid: type(internal_model[int(lid)]).__name__ for lid in self.internal_ids}  # type: ignore[index]
            source_recording = Recording(
                RecordingType.SOURCE, source_recordings,
                layer_types=layer_types, internal_recording=internal_recording,
            )
            self.stages[RecordingType.SOURCE] = NetworkStage(
                name="Source",
                parameter_space=Real(),
                recordings=source_recording,
            )

        # QUANTIZED stage
        quantized_model, quant_data = self._quantize_model(internal_model)
        quantized_recordings = self._execute_network(
            quantized_model, self.data_sample, self.single_save_hook
        )
        quantized_recording = Recording(RecordingType.QUANTIZED, quantized_recordings)
        self.stages[RecordingType.QUANTIZED] = NetworkStage(
            name="Quantized",
            parameter_space=Quantized(properties=quant_data),
            recordings=quantized_recording,
            model=quantized_model,
        )

    def get_dataloader(self, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.dc.dataset,
            batch_size=self.dc.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def get_representative_sample(self) -> tuple[torch.Tensor, Any]:
        dataset = self.dc.dataset
        index = max(0, min(self.dc.representative_sample_index, len(cast(Any, dataset)) - 1))
        data, label = dataset[index]
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data).float()
        return data, label

    def _materialize_calibration_dataset(self) -> Dataset:
        dataset = self.dc.effective_calibration_dataset()
        assert self.dc.ptq is not None
        num_samples = self.dc.ptq.num_samples
        if num_samples is None or num_samples >= len(cast(Any, dataset)):
            return dataset

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.dc.ptq.seed)
        indices = torch.randperm(len(cast(Any, dataset)), generator=generator)[:num_samples].tolist()
        return Subset(dataset, indices)

    def get_calibration_dataloader(self) -> DataLoader:
        return DataLoader(
            self._materialize_calibration_dataset(),
            batch_size=self.dc.effective_calibration_batch_size(),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    @staticmethod
    def _merge_min_max_recordings(
        target: Dict[str, Dict[str, torch.Tensor]],
        source: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        for layer_id, layer_recording in source.items():
            if layer_id in ("input", "output"):
                continue
            if layer_id not in target:
                target[layer_id] = {
                    key: value.clone().detach() for key, value in layer_recording.items()
                }
                continue

            for key, value in layer_recording.items():
                if key not in target[layer_id]:
                    target[layer_id][key] = value.clone().detach()
                    continue
                target[layer_id][key][0] = torch.min(target[layer_id][key][0], value[0])
                target[layer_id][key][1] = torch.max(target[layer_id][key][1], value[1])
        return target

    def _init_histogram_state(
        self,
        min_max_recordings: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        assert self.dc.ptq is not None
        bins = self.dc.ptq.histogram_bins
        if bins <= 1:
            raise ValueError(f"histogram_bins must be > 1, got {bins}")

        state: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for layer_id, layer_recording in min_max_recordings.items():
            state[layer_id] = {}
            for param_name, value in layer_recording.items():
                min_val = float(value[0].item())
                max_val = float(value[1].item())
                state[layer_id][param_name] = {
                    "min": min_val,
                    "max": max_val,
                    "counts": torch.zeros(bins, dtype=torch.float64),
                }
        return state

    @staticmethod
    def _update_histogram_counts(
        histogram_state: Dict[str, Dict[str, Dict[str, Any]]],
        layer_id: str,
        param_name: str,
        values: torch.Tensor,
        sample_stride: int,
    ) -> None:
        if layer_id not in histogram_state or param_name not in histogram_state[layer_id]:
            return

        entry = histogram_state[layer_id][param_name]
        counts: torch.Tensor = entry["counts"]
        min_val = entry["min"]
        max_val = entry["max"]
        bins = counts.numel()

        vals = values.detach().flatten().float()
        if vals.numel() == 0:
            return
        if sample_stride > 1:
            vals = vals[::sample_stride]
            if vals.numel() == 0:
                return

        if max_val <= min_val:
            counts[0] += float(vals.numel())
            return

        # Map values into fixed bins to keep memory bounded by O(bins).
        norm = (vals - min_val) / (max_val - min_val)
        idx = torch.clamp(torch.floor(norm * bins).long(), 0, bins - 1)
        bincount = torch.bincount(idx.cpu(), minlength=bins).to(dtype=torch.float64)
        counts += bincount

    @staticmethod
    def _validate_normalized_nir_graph(nir_graph: nir.NIRGraph) -> None:
        node_ids = set(nir_graph.nodes.keys())
        if "input" not in node_ids or "output" not in node_ids:
            raise ValueError(
                "Normalized NIR graph must contain canonical 'input' and 'output' nodes."
            )

        internal_ids = [nid for nid in node_ids if nid not in {"input", "output"}]
        if any(not nid.isdigit() for nid in internal_ids):
            raise ValueError(
                "Normalized NIR graph must name every non-boundary node with a numeric id."
            )

        ordered_ids = sorted(internal_ids, key=int)
        if ordered_ids != [str(i) for i in range(len(ordered_ids))]:
            raise ValueError(
                "Normalized NIR graph must use contiguous numeric node ids starting at 0; "
                f"got {ordered_ids}"
            )

        expected_chain = ["input", *ordered_ids, "output"]
        expected_edges = set(zip(expected_chain, expected_chain[1:]))
        if set(nir_graph.edges) != expected_edges:
            raise ValueError(
                "Normalized NIR graph must be a single chain input → 0 → 1 → … → output; "
                f"expected edges {expected_edges}, got {set(nir_graph.edges)}"
            )

        input_node = cast(Any, nir_graph.nodes["input"])
        for key, value in input_node.input_type.items():
            arr = np.array(value)
            if arr.ndim != 1 or len(arr) != 1:
                raise ValueError(
                    f"Input shape '{key}' must be 1D after normalization, got {arr.tolist()!r}. "
                    "Multi-dimensional inputs are not supported."
                )

    def _histogram_hook(
        self,
        histogram_state: Dict[str, Dict[str, Dict[str, Any]]],
        sample_stride: int,
    ) -> Callable[[str, str, int, int, Dict], Callable]:
        def hook_factory(
            layer_type: str,
            layer_id: str,
            batch_size: int,
            timesteps: int,
            recordings: Dict[str, Dict[str, torch.Tensor]],
        ) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
            match layer_type:
                case "IAFSqueeze" | "LIFSqueeze":
                    def neuron_hook(mod: torch.nn.Module, i: tuple, o: torch.Tensor, layer_id: str = layer_id) -> None:
                        in_tensor = i[0].detach()
                        out_tensor = o.detach()
                        v_mem = mod.recordings["v_mem"].detach()  # type: ignore[union-attr]
                        i_syn = (
                            mod.recordings["i_syn"].detach()  # type: ignore[union-attr]
                            if "i_syn" in mod.recordings.keys()  # type: ignore[union-attr]
                            else torch.zeros_like(v_mem)
                        )
                        self._update_histogram_counts(histogram_state, layer_id, "input", in_tensor, sample_stride)
                        self._update_histogram_counts(histogram_state, layer_id, "output", out_tensor, sample_stride)
                        self._update_histogram_counts(histogram_state, layer_id, "v_mem", v_mem, sample_stride)
                        self._update_histogram_counts(histogram_state, layer_id, "i_syn", i_syn, sample_stride)

                    return neuron_hook
                case "ExpLeakSqueeze" | "CurrentFilterSqueeze":
                    def leak_hook(mod: torch.nn.Module, i: tuple, o: torch.Tensor, layer_id: str = layer_id) -> None:
                        in_tensor = i[0].detach()
                        out_tensor = o.detach()
                        self._update_histogram_counts(histogram_state, layer_id, "input", in_tensor, sample_stride)
                        self._update_histogram_counts(histogram_state, layer_id, "output", out_tensor, sample_stride)
                        self._update_histogram_counts(histogram_state, layer_id, "v_mem", out_tensor, sample_stride)

                    return leak_hook
                case "Linear":
                    def linear_hook(mod: torch.nn.Module, i: tuple, o: torch.Tensor, layer_id: str = layer_id) -> None:
                        in_tensor = i[0].detach()
                        out_tensor = o.detach()
                        w = mod.weight.detach()  # type: ignore[union-attr]
                        self._update_histogram_counts(histogram_state, layer_id, "input", in_tensor, sample_stride)
                        self._update_histogram_counts(histogram_state, layer_id, "output", out_tensor, sample_stride)
                        self._update_histogram_counts(histogram_state, layer_id, "weights", w, sample_stride)

                    return linear_hook
                case _:
                    raise NotImplementedError(
                        f'NIR2FPGA: Layer "{layer_type}" does not have histogram hook implemented.'
                    )

        return hook_factory

    def _histogram_percentile_bounds(
        self,
        histogram_state: Dict[str, Dict[str, Dict[str, Any]]],
        percentile: float,
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        if not 0.0 < percentile <= 100.0:
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")

        tail = (100.0 - percentile) / 2.0
        lower_q = tail / 100.0
        upper_q = 1.0 - lower_q

        bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for layer_id, layer_state in histogram_state.items():
            bounds[layer_id] = {}
            for param_name, entry in layer_state.items():
                min_val = float(entry["min"])
                max_val = float(entry["max"])
                counts = cast(torch.Tensor, entry["counts"])
                total = float(counts.sum().item())

                if total <= 0.0 or max_val <= min_val:
                    bounds[layer_id][param_name] = (min_val, max_val)
                    continue

                cdf = torch.cumsum(counts, dim=0)
                lower_target = lower_q * total
                upper_target = upper_q * total
                lower_idx = int(torch.searchsorted(cdf, torch.tensor(lower_target, dtype=cdf.dtype), right=False).item())
                upper_idx = int(torch.searchsorted(cdf, torch.tensor(upper_target, dtype=cdf.dtype), right=False).item())
                lower_idx = max(0, min(lower_idx, counts.numel() - 1))
                upper_idx = max(0, min(upper_idx, counts.numel() - 1))

                bin_width = (max_val - min_val) / counts.numel()
                lower_val = min_val + (lower_idx * bin_width)
                upper_val = min_val + ((upper_idx + 1) * bin_width)
                lower_val = max(min_val, min(lower_val, max_val))
                upper_val = max(min_val, min(upper_val, max_val))
                if upper_val < lower_val:
                    upper_val = lower_val

                bounds[layer_id][param_name] = (lower_val, upper_val)

        return bounds

    def _collect_percentile_bounds(
        self,
        model: torch.nn.Sequential,
        min_max_recordings: Dict[str, Dict[str, torch.Tensor]],
        percentile: float,
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        histogram_state = self._init_histogram_state(min_max_recordings)
        assert self.dc.ptq is not None
        sample_stride = max(1, self.dc.ptq.histogram_sample_stride)

        hook = self._histogram_hook(histogram_state, sample_stride)
        cal_loader = self.get_calibration_dataloader()
        total_batches = len(cal_loader)
        for i, (batch_x, _) in enumerate(cal_loader):
            if not isinstance(batch_x, torch.Tensor):
                batch_x = torch.tensor(batch_x).float()
            _ = self._execute_network(model, batch_x.float(), hook)
            print(f"\rCalibrating (histogram): {i + 1}/{total_batches}", end="", flush=True)
        print()

        return self._histogram_percentile_bounds(histogram_state, percentile)

    def _execute_network(self,
                        network: torch.nn.Sequential,
                        data: torch.Tensor,
                        hooks: Callable[[str, str, int, int, Dict], Callable],
                        preps: Optional[Callable[[str, torch.nn.Module], None]] = None
                        ) -> Dict[str, Any]:

        if preps is None:
            preps = self.module_prepare
        eval_model = copy.deepcopy(network)
        for module in eval_model.modules():
            if hasattr(module, "reset_states"):
                module.reset_states()  # type: ignore[operator]

        recordings: Dict[str, Any] = {layer_id: {} for layer_id in self.internal_ids}
        batch_size = data.shape[0]
        timesteps = self.dc.timesteps

        with torch.no_grad():
            x = data.flatten(0, 1)
            for layer_id, layer in zip(self.internal_ids, eval_model):
                cls_name = type(layer).__name__
                if cls_name == "QuantizationWrapper":
                    cls_name = layer.layer_name
                preps(cls_name, layer)  # type: ignore[arg-type]
                hook_fn = hooks(cls_name, layer_id, batch_size, timesteps, recordings)  # type: ignore[arg-type]
                handle = layer.register_forward_hook(hook_fn)
                x = layer(x)
                handle.remove()
            output = x.view(data.shape[0], data.shape[1], *x.shape[1:])

        recordings["input"] = {"output": data}
        recordings["output"] = {"input": output}

        return recordings



    def add_recording(self, stage_type: RecordingType, recording_dict: Dict[str, Dict[str, torch.Tensor]],
                      observed_accuracy: Optional[float] = None) -> None:
        allowed = {RecordingType.SOURCE, RecordingType.BEHAVIOURAL, RecordingType.HARDWARE}
        if stage_type not in allowed:
            raise ValueError(
                f"stage_type must be one of {[s.name for s in allowed]}, got '{stage_type.name}'"
            )

        if observed_accuracy is not None and stage_type != RecordingType.SOURCE:
            import warnings
            warnings.warn(
                f"observed_accuracy is only meaningful for SOURCE recordings; "
                f"ignoring for {stage_type.name}.",
                UserWarning,
            )

        stage_to_rec = {
            RecordingType.SOURCE: RecordingType.SOURCE,
            RecordingType.BEHAVIOURAL: RecordingType.BEHAVIOURAL,
            RecordingType.HARDWARE: RecordingType.HARDWARE,
        }

        internal_model = self.stages[RecordingType.INTERNAL].model
        assert internal_model is not None
        layer_types = {lid: type(internal_model[int(lid)]).__name__ for lid in self.internal_ids}  # type: ignore[index]

        # Merge with existing recordings so repeated calls append rather than overwrite
        existing_stage = self.stages.get(stage_type)
        if existing_stage is not None and existing_stage.recordings is not None:
            merged = dict(existing_stage.recordings.data)
            merged.update(recording_dict)
            recording_dict = merged

        rec = Recording(
            stage_to_rec[stage_type], recording_dict,
            layer_types=layer_types,
            internal_recording=self.stages[RecordingType.INTERNAL].recordings,
        )

        self.stages[stage_type] = NetworkStage(
            name=stage_type.name.capitalize(),
            parameter_space=Real(),
            recordings=rec,
            accuracy=observed_accuracy if stage_type == RecordingType.SOURCE else None,
        )

    @staticmethod
    def _optional_node_scalar(node: Any, field: str) -> Optional[float]:
        if not hasattr(node, field):
            return None
        value = getattr(node, field)
        return float(np.asarray(value).reshape(-1)[0])
    @staticmethod
    def _node_output_shape(node: Any) -> Optional[torch.Size]:
        if hasattr(node, "output_type") and "output" in getattr(node, "output_type"):
            output_shape = np.asarray(node.output_type["output"]).tolist()
            return torch.Size(output_shape)
        if hasattr(node, "shape"):
            shape_value = np.asarray(getattr(node, "shape")).tolist()
            return torch.Size(shape_value)
        return None

    def _ordered_nir_nodes(self, nir_graph: nir.NIRGraph) -> list[tuple[str, Any]]:
        numeric_items = [(node_id, node) for node_id, node in nir_graph.nodes.items() if node_id.isdigit()]
        if not numeric_items:
            return []

        ordered_ids = [node_id for node_id, _ in sorted(numeric_items, key=lambda item: int(item[0]))]
        expected_ids = [str(index) for index in range(len(ordered_ids))]
        if ordered_ids != expected_ids:
            raise ValueError(
                "Normalized NIR graph must be a single ordered chain with contiguous numeric node ids; "
                f"got {ordered_ids}"
            )

        return [(node_id, nir_graph.nodes[node_id]) for node_id in ordered_ids]

    def _build_layer_from_nir_node(self, node_id: str, node: Any) -> Optional[torch.nn.Module]:
        node_type = type(node).__name__

        if node_type in {"Input", "Output", "Identity"}:
            return None

        if node_type == "Flatten":
            return torch.nn.Flatten()

        if node_type in {"Affine", "Linear"}:
            weight = torch.tensor(np.asarray(node.weight), dtype=torch.float32)
            bias = getattr(node, "bias", None)
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
            layer.weight.data.copy_(weight)
            if bias is not None:
                layer.bias.data.copy_(torch.tensor(np.asarray(bias), dtype=torch.float32))
            return layer

        neuron_shape = self._node_output_shape(node)
        spike_threshold = self._optional_node_scalar(node, "v_threshold")
        min_v_mem = self._optional_node_scalar(node, "min_v_mem")

        if node_type == "IF":
            return sinabs.layers.IAFSqueeze(  # type: ignore[attr-defined]
                spike_threshold=torch.tensor(1.0 if spike_threshold is None else spike_threshold, dtype=torch.float32),
                spike_fn=self.conversion_params["IAFSqueeze"]["spike_fn"],
                reset_fn=self.conversion_params["IAFSqueeze"]["reset_fn"],
                tau_syn=self._optional_node_scalar(node, "tau_syn"),
                min_v_mem=min_v_mem,
                shape=neuron_shape,
                num_timesteps=self.dc.timesteps,
            )

        if node_type == "LIF":
            tau_mem = float(np.asarray(node.tau).reshape(-1)[0])
            tau_syn = self._optional_node_scalar(node, "tau_syn")
            return sinabs.layers.LIFSqueeze(  # type: ignore[attr-defined]
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                spike_threshold=torch.tensor(1.0 if spike_threshold is None else spike_threshold, dtype=torch.float32),
                spike_fn=self.conversion_params["LIFSqueeze"]["spike_fn"],
                reset_fn=self.conversion_params["LIFSqueeze"]["reset_fn"],
                min_v_mem=min_v_mem,
                train_alphas=getattr(node, "train_alphas", False),
                shape=neuron_shape,
                norm_input=self.conversion_params["LIFSqueeze"]["norm_input"],
                num_timesteps=self.dc.timesteps,
            )

        if node_type == "LI":
            tau_mem = float(np.asarray(node.tau).reshape(-1)[0])
            return CurrentFilterSqueeze(
                tau_mem=tau_mem,
                min_v_mem=min_v_mem,
                train_alphas=getattr(node, "train_alphas", False),
                shape=neuron_shape,
                norm_input=self.conversion_params["ExpLeakSqueeze"]["norm_input"],
                num_timesteps=self.dc.timesteps,
            )

        if node_type == "I":
            return IntegratorSqueeze(
                tau_mem=1e30,
                min_v_mem=min_v_mem,
                shape=neuron_shape,
                num_timesteps=self.dc.timesteps,
            )

        if node_type == "CuBaLIF":
            raise NotImplementedError(
                f"The framework currently doesn't support the CuBaLIF primitive directly. Please decompose into NIR: LI -> Linear -> LIF"
            )

        raise NotImplementedError(
            f'NIR2FPGA: Layer "{node_id}" ({node_type}) is not supported for direct construction yet.'
        )

    def _nir_to_internal(self, nir_graph):
        """
        Converts the NIR graph into our internal simulator.
        The NIR graph must not contain nodes that are not listed here: https://neuroir.org/docs/examples/sinabs/nir-conversion.html

        Internal simulator: nn.Sequential with {torch, sinabs} modules.
        """
        sequential_model: list[torch.nn.Module] = []
        self.nir_name_to_internal_id: Dict[str, str] = {}
        seq_idx = 0
        for node_id, node in self._ordered_nir_nodes(nir_graph):
            layer = self._build_layer_from_nir_node(node_id, node)
            if layer is None:
                continue
            sequential_model.append(layer)
            self.nir_name_to_internal_id[node_id] = str(seq_idx)
            seq_idx += 1
        self.internal_id_to_nir_node: Dict[str, Any] = {
            internal_id: nir_graph.nodes[nir_node_id]
            for nir_node_id, internal_id in self.nir_name_to_internal_id.items()
        }
        return torch.nn.Sequential(*sequential_model)

    def _quantize_model(self, model: torch.nn.Sequential):
        assert self.dc.ptq is not None
        if self.dc.ptq.method not in {"minmax", "percentile"}:
            raise NotImplementedError(
                f"Calibration method '{self.dc.ptq.method}' is not implemented. Supported methods: ['minmax', 'percentile']"
            )

        recordings: Dict[str, Dict[str, torch.Tensor]] = {}
        cal_loader = self.get_calibration_dataloader()
        total_batches = len(cal_loader)
        for i, (batch_x, _) in enumerate(cal_loader):
            if not isinstance(batch_x, torch.Tensor):
                batch_x = torch.tensor(batch_x).float()
            batch_recordings = self._execute_network(model, batch_x.float(), self.min_max_hook)
            recordings = self._merge_min_max_recordings(recordings, batch_recordings)
            print(f"\rCalibrating (min-max): {i + 1}/{total_batches}", end="", flush=True)
        print()

        percentile_bounds_by_layer: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
        readout_percentile_bounds_by_layer: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
        if self.dc.ptq.method == "percentile" or self.dc.ptq.readout_percentile is not None:
            cal_percentile = 99.9 if self.dc.ptq.percentile is None else self.dc.ptq.percentile
            percentile_bounds_by_layer = self._collect_percentile_bounds(model, recordings, cal_percentile)
            if self.dc.ptq.readout_percentile is not None:
                readout_percentile_bounds_by_layer = self._collect_percentile_bounds(
                    model, recordings, self.dc.ptq.readout_percentile
                )

        _NIR_TYPE_TO_SINABS_KEY: Dict[str, str] = {
            "IF": "IAFSqueeze",
            "LIF": "LIFSqueeze",
            "CubaLIF": "LIFSqueeze",
            "LI": "ExpLeakSqueeze",
            "I": "ExpLeakSqueeze",
            "Linear": "Linear",
            "Affine": "Linear",
        }

        quantized_model = []
        quant_data = {}
        assert self.dc.ptq is not None
        for layer_id, _ in zip(self.internal_ids, model):
            nir_node = self.internal_id_to_nir_node[layer_id]
            nir_type = type(nir_node).__name__
            sinabs_key = _NIR_TYPE_TO_SINABS_KEY.get(nir_type, "")
            spike_fn = self.conversion_params.get(sinabs_key, {}).get("spike_fn", None)
            norm_input = self.conversion_params.get(sinabs_key, {}).get("norm_input", False)
            try:
                quant = QuantizationConfig(
                    recordings[layer_id],
                    nir_node,
                    self.dc.total_bits,
                    weight_bits=self.dc.weight_bits,
                    calibration_method=self.dc.ptq.method,
                    calibration_percentile=self.dc.ptq.percentile,
                    threshold_headroom_multiplier=self.dc.ptq.threshold_headroom_multiplier,
                    readout_percentile=self.dc.ptq.readout_percentile,
                    percentile_bounds=(
                        percentile_bounds_by_layer[layer_id]
                        if percentile_bounds_by_layer is not None and layer_id in percentile_bounds_by_layer
                        else None
                    ),
                    readout_percentile_bounds=(
                        readout_percentile_bounds_by_layer[layer_id]
                        if readout_percentile_bounds_by_layer is not None and layer_id in readout_percentile_bounds_by_layer
                        else None
                    ),
                    spike_fn=spike_fn,
                    norm_input=norm_input,
                    num_timesteps=self.dc.timesteps,
                )
            except ValueError as exc:
                raise ValueError(f"Layer {layer_id} ({nir_type}) quantization failed: {exc}") from exc
            quant_data[layer_id] = quant
            quantized_model.append(quant.wrapper)

        quantized_model = torch.nn.Sequential(*quantized_model)

        # Fill in input and output quantizations
        quant_data["input"] = {"output": quant_data[self.internal_ids[0]]["input"]}
        quant_data["output"] = {"input": quant_data[self.internal_ids[-1]]["output"]}

        return (quantized_model, quant_data)

    @staticmethod
    def module_prepare(
            layer_type: str,
            module: torch.nn.Module
    ) -> None:
        match layer_type:
            case "IAFSqueeze" | "LIFSqueeze" | "IF" | "LIF" | "CubaLIF":
                module.record_states = True  # type: ignore[assignment]
            case "ExpLeakSqueeze" | "CurrentFilterSqueeze" | "LI" | "IntegratorSqueeze" | "I":
                pass  # No internal state recording needed; output == v_mem for LI
            case "Linear" | "Affine":
                pass
            case _:
                raise NotImplementedError(
                    f'NIR2FPGA: Layer "{layer_type}" does not have preparation method implemented.'
                )

    @staticmethod
    def single_save_hook(
            layer_type: str,
            layer_id: str,
            batch_size: int,
            timesteps: int,
            recordings: Dict[str, Dict[str, torch.Tensor]],
    ) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
        returned_hook = None
        match layer_type:
            case "IAFSqueeze" | "LIFSqueeze" | "IF" | "LIF" | "CubaLIF":
                def activation_hook(mod, i, o, layer_id=layer_id):
                    inputs = i[0].clone().detach()
                    inputs = inputs.view(batch_size, timesteps, *inputs.shape[1:])
                    outputs = o.clone().detach()
                    outputs = outputs.view(batch_size, timesteps, *outputs.shape[1:])
                    # squeeze(0) debatches: (1, time, neurons) → (time, neurons)
                    v_mem = mod.recordings["v_mem"].clone().detach().squeeze(0)
                    i_syn = (
                        mod.recordings["i_syn"].clone().detach().squeeze(0)
                        if "i_syn" in mod.recordings.keys()
                        else torch.zeros_like(v_mem)
                    )

                    recordings[layer_id]["input"] = inputs[0]
                    recordings[layer_id]["output"] = outputs[0]
                    recordings[layer_id]["v_mem"] = v_mem
                    recordings[layer_id]["i_syn"] = i_syn

                returned_hook = activation_hook
            case "Linear" | "Affine" | "ExpLeakSqueeze" | "CurrentFilterSqueeze" | "LI" | "IntegratorSqueeze" | "I":
                def activation_hook(mod, i, o, layer_id=layer_id):
                    inputs = i[0].detach()
                    inputs = inputs.view(batch_size, timesteps, *inputs.shape[1:])
                    outputs = o.detach()
                    outputs = outputs.view(batch_size, timesteps, *outputs.shape[1:])
                    recordings[layer_id]["input"] = inputs[0]
                    recordings[layer_id]["output"] = outputs[0]

                returned_hook = activation_hook
            case _:
                raise NotImplementedError(
                    f'NIR2FPGA: Layer "{layer_type}" does not have evaluation hook implemented.'
                )

        return returned_hook

    @staticmethod
    def min_max_hook(
            layer_type: str,
            layer_id: str,
            batch_size: int,
            timesteps: int,
            recordings: Dict[str, Dict[str, torch.Tensor]],
    ) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
        returned_hook = None
        match layer_type:
            case "IAFSqueeze" | "LIFSqueeze" | "IF" | "LIF" | "CubaLIF":
                def activation_hook(mod, i, o, layer_id=layer_id):  # type: ignore[reportRedeclaration]
                    i = i[0].detach()
                    o = o.detach()
                    v_mem = mod.recordings["v_mem"].detach()
                    i_syn = (
                        mod.recordings["i_syn"][0].detach()
                        if "i_syn" in mod.recordings.keys()
                        else torch.zeros_like(v_mem)
                    )

                    if len(recordings[layer_id].keys()) == 0:
                        recordings[layer_id]["input"] = torch.stack([i.min(), i.max()])
                        recordings[layer_id]["output"] = torch.stack([o.min(), o.max()])
                        recordings[layer_id]["v_mem"] = torch.stack([v_mem.min(), v_mem.max()])
                        recordings[layer_id]["i_syn"] = torch.stack([i_syn.min(), i_syn.max()])
                    else:
                        recordings[layer_id]["input"][0] = torch.min(recordings[layer_id]["input"][0], i.min())
                        recordings[layer_id]["input"][1] = torch.max(recordings[layer_id]["input"][1], i.max())
                        recordings[layer_id]["output"][0] = torch.min(recordings[layer_id]["output"][0], o.min())
                        recordings[layer_id]["output"][1] = torch.max(recordings[layer_id]["output"][1], o.max())
                        recordings[layer_id]["v_mem"][0] = torch.min(recordings[layer_id]["v_mem"][0], v_mem.min())
                        recordings[layer_id]["v_mem"][1] = torch.max(recordings[layer_id]["v_mem"][1], v_mem.max())
                        recordings[layer_id]["i_syn"][0] = torch.min(recordings[layer_id]["i_syn"][0], i_syn.min())
                        recordings[layer_id]["i_syn"][1] = torch.max(recordings[layer_id]["i_syn"][1], i_syn.max())

                returned_hook = activation_hook
            case "ExpLeakSqueeze" | "CurrentFilterSqueeze" | "LI" | "IntegratorSqueeze" | "I":
                def activation_hook(mod, i, o, layer_id=layer_id):  # type: ignore[reportRedeclaration]
                    i = i[0].detach()
                    o = o.detach()
                    # For LI, output == v_mem; alias v_mem to output for quantization compatibility
                    if len(recordings[layer_id].keys()) == 0:
                        recordings[layer_id]["input"] = torch.stack([i.min(), i.max()])
                        recordings[layer_id]["output"] = torch.stack([o.min(), o.max()])
                        recordings[layer_id]["v_mem"] = torch.stack([o.min(), o.max()])
                    else:
                        recordings[layer_id]["input"][0] = torch.min(recordings[layer_id]["input"][0], i.min())
                        recordings[layer_id]["input"][1] = torch.max(recordings[layer_id]["input"][1], i.max())
                        recordings[layer_id]["output"][0] = torch.min(recordings[layer_id]["output"][0], o.min())
                        recordings[layer_id]["output"][1] = torch.max(recordings[layer_id]["output"][1], o.max())
                        recordings[layer_id]["v_mem"][0] = torch.min(recordings[layer_id]["v_mem"][0], o.min())
                        recordings[layer_id]["v_mem"][1] = torch.max(recordings[layer_id]["v_mem"][1], o.max())

                returned_hook = activation_hook
            case "Linear" | "Affine":
                def activation_hook(mod, i, o, layer_id=layer_id):
                    i = i[0].detach()
                    o = o.detach()
                    w = mod.weight.clone().detach()

                    if len(recordings[layer_id].keys()) == 0:
                        recordings[layer_id]["input"] = torch.stack([i.min(), i.max()])
                        recordings[layer_id]["output"] = torch.stack([o.min(), o.max()])
                        recordings[layer_id]["weights"] = torch.stack([w.min(), w.max()])

                    else:
                        recordings[layer_id]["input"][0] = torch.min(recordings[layer_id]["input"][0], i.min())
                        recordings[layer_id]["input"][1] = torch.max(recordings[layer_id]["input"][1], i.max())
                        recordings[layer_id]["output"][0] = torch.min(recordings[layer_id]["output"][0], o.min())
                        recordings[layer_id]["output"][1] = torch.max(recordings[layer_id]["output"][1], o.max())
                        recordings[layer_id]["weights"][0] = torch.min(recordings[layer_id]["weights"][0], w.min())
                        recordings[layer_id]["weights"][1] = torch.max(recordings[layer_id]["weights"][1], w.max())

                returned_hook = activation_hook
            case _:
                raise NotImplementedError(
                    f'NIR2FPGA: Layer "{layer_type}" does not have evaluation hook implemented.'
                )
        return returned_hook  # type: ignore[return-value]

    @staticmethod
    def output_save_hook(
            layer_type: str,
            layer_id: str,
            batch_size: int,
            timesteps: int,
            recordings: Dict[str, Dict[str, torch.Tensor]],
    ) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
        def activation_hook(mod, i, o, layer_id=layer_id):
            outputs = o.clone().detach()
            outputs = outputs.view(batch_size, timesteps, *outputs.shape[1:])
            if "output" not in recordings[layer_id].keys():
                recordings[layer_id]["output"] = outputs
            else:
                recordings[layer_id]["output"] = torch.cat([recordings[layer_id]["output"], outputs], dim=0)

        return activation_hook

    @staticmethod
    def targeted_save_hook(
        target_layer_id: str,
        target_param: str,
    ) -> Callable[[str, str, int, int, Dict], Callable]:
        def hook_factory(
            layer_type: str,
            layer_id: str,
            batch_size: int,
            timesteps: int,
            recordings: Dict[str, Dict[str, torch.Tensor]],
        ) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
            if layer_id != target_layer_id:
                def noop(mod, i, o):
                    pass
                return noop
            match target_param:
                case "output":
                    def activation_hook(mod, i, o, layer_id=layer_id):
                        outputs = o.clone().detach()
                        recordings[layer_id]["output"] = outputs.view(
                            batch_size, timesteps, *outputs.shape[1:]
                        )
                    return activation_hook
                case "input":
                    def activation_hook(mod, i, o, layer_id=layer_id):
                        inputs = i[0].clone().detach()
                        recordings[layer_id]["input"] = inputs.view(
                            batch_size, timesteps, *inputs.shape[1:]
                        )
                    return activation_hook
                case "v_mem":
                    def activation_hook(mod, i, o, layer_id=layer_id):
                        # sinabs already stores v_mem as (batch, timesteps, neurons)
                        recordings[layer_id]["v_mem"] = mod.recordings["v_mem"].clone().detach()
                    return activation_hook
                case "i_syn":
                    def activation_hook(mod, i, o, layer_id=layer_id):
                        # sinabs already stores i_syn as (batch, timesteps, neurons)
                        recordings[layer_id]["i_syn"] = mod.recordings["i_syn"].clone().detach()
                    return activation_hook
                case _:
                    raise NotImplementedError(
                        f"targeted_save_hook: parameter '{target_param}' is not supported."
                    )
        return hook_factory

    @staticmethod
    def _is_jupyter() -> bool:
        try:
            from IPython import get_ipython  # type: ignore[attr-defined]
            shell = get_ipython()
            return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
        except ImportError:
            return False

    def evaluate_stages(self, prediction_fn: PredictionType) -> List[Tuple[RecordingType, float]]:
        stages_sorted = sorted(list(self.stages.keys()))
        if isinstance(prediction_fn, (OutputEqual, LayerEqual)):
            return self._evaluate_output_equal_cascade(stages_sorted, prediction_fn)
        return self._evaluate_standard(stages_sorted, prediction_fn)

    def evaluate(self, prediction_fn: PredictionType):
        accuracies = self.evaluate_stages(prediction_fn)

        print("|{:>10}|{:>10}|".format("Stage", "Accuracy"))
        print("+{}+{}+".format("-"*10, "-"*10))
        for (stage_type, acc) in accuracies:
            print("|{:>10}|{:>10}|".format(stage_type, f"{acc * 100:.2f}%"))

    def _evaluate_standard(self,
            stages_sorted: List[RecordingType],
            prediction_fn: PredictionType
    ) -> List[Tuple[RecordingType, float]]:
        accuracies: List[Tuple[RecordingType, float]] = []
        for stage_type in stages_sorted:
            if stage_type not in [RecordingType.SOURCE, RecordingType.INTERNAL, RecordingType.QUANTIZED]:
                continue

            stage = self.stages[stage_type]
            accuracy: float = 0.0
            match stage_type:
                case RecordingType.SOURCE:
                    if stage.accuracy is not None:
                        accuracy = stage.accuracy
                    else:
                        accuracy = prediction_fn(stage.recordings, self.labels)
                case RecordingType.INTERNAL | RecordingType.QUANTIZED:
                    total_correct = 0.0
                    total_samples = 0
                    dataloader = self.get_dataloader()
                    total_batches = len(dataloader)
                    for i, (x, y) in enumerate(dataloader):
                        assert stage.model is not None
                        rec = self._execute_network(stage.model, x.float(), self.output_save_hook)  # type: ignore[arg-type]
                        batch_recording = Recording(stage_type, rec)
                        total_correct += prediction_fn(batch_recording, y) * x.shape[0]
                        total_samples += x.shape[0]
                        print(f"\rEvaluating {stage.name}: {i + 1}/{total_batches}", end="", flush=True)
                    print()
                    accuracy = total_correct / total_samples

            accuracies.append((stage_type, accuracy))
        return accuracies

    def _evaluate_output_equal_cascade(
            self, stages_sorted: List[RecordingType], prediction_fn: PredictionType
    ) -> List[Tuple[RecordingType, float]]:
        assert isinstance(prediction_fn, (OutputEqual, LayerEqual))

        if isinstance(prediction_fn, OutputEqual):
            target_layer, target_param = "output", "input"
            atol, rtol = 0.0, 0.0
        else:  # LayerEqual
            target_layer = prediction_fn.layer
            target_param = prediction_fn.parameter
            atol, rtol   = prediction_fn.atol, prediction_fn.rtol

        hook = self.targeted_save_hook(target_layer, target_param)

        present = set(stages_sorted)

        # Each stage's parent in the cascade
        parents: Dict[RecordingType, RecordingType] = {
            RecordingType.INTERNAL:    RecordingType.SOURCE,
            RecordingType.QUANTIZED:   RecordingType.INTERNAL,
            RecordingType.BEHAVIOURAL: RecordingType.QUANTIZED,
            RecordingType.HARDWARE:    RecordingType.QUANTIZED,
        }

        # Stages whose parent is absent become roots → 100%
        root_stages = {
            st for st, par in parents.items()
            if st in present and par not in present
        }
        if RecordingType.SOURCE in present:
            root_stages.add(RecordingType.SOURCE)

        stage_correct:  Dict[RecordingType, float] = {st: 0.0 for st in present}
        stage_elements: Dict[RecordingType, int]   = {st: 0   for st in present}

        for batch_idx, (x, _) in enumerate(self.get_dataloader()):
            batch_start = batch_idx * self.dc.batch_size
            batch_end   = batch_start + x.shape[0]

            batch_out: Dict[RecordingType, torch.Tensor] = {}

            if RecordingType.SOURCE in present:
                batch_out[RecordingType.SOURCE] = (
                    self.stages[RecordingType.SOURCE]
                        .recordings.data[target_layer][target_param][batch_start:batch_end]
                )
            for model_stage in (RecordingType.INTERNAL, RecordingType.QUANTIZED):
                if model_stage in present:
                    rec = self._execute_network(
                        self.stages[model_stage].model, x.float(), hook  # type: ignore[arg-type]
                    )
                    batch_out[model_stage] = rec[target_layer][target_param]
            for rec_stage in (RecordingType.BEHAVIOURAL, RecordingType.HARDWARE):
                if rec_stage in present:
                    batch_out[rec_stage] = (
                        self.stages[rec_stage]
                            .recordings.data[target_layer][target_param][batch_start:batch_end]
                    )

            for stage_type, parent_type in parents.items():
                if stage_type not in present or stage_type in root_stages:
                    continue
                if parent_type not in batch_out or stage_type not in batch_out:
                    continue
                correct = torch.isclose(
                    batch_out[stage_type], batch_out[parent_type],
                    atol=atol, rtol=rtol,
                )
                stage_correct[stage_type]  += correct.float().sum().item()
                stage_elements[stage_type] += correct.numel()

        accuracies: List[Tuple[RecordingType, float]] = []
        for st in stages_sorted:
            if st in root_stages:
                accuracies.append((st, 1.0))
            elif stage_elements[st] > 0:
                accuracies.append((st, stage_correct[st] / stage_elements[st]))
        return accuracies
