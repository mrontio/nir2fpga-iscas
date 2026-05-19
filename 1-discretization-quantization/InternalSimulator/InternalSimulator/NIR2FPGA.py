import copy
import math
import nir
import torch
import re
import json
import sinabs.activation
from pathlib import Path
import time
import numpy as np
from typing import Any, Dict, List, Optional, cast

from collections.abc import Sized

from InternalSimulator.DiscretizationChoices import DiscretizationChoices
from InternalSimulator.HardwareSimulation import HardwareSimulation, SimulationOptions
from InternalSimulator.Quantization import QuantizationConfig, QuantizationWrapper
from InternalSimulator.IOManager import IOManager
from InternalSimulator.Plotter import Plotter
from InternalSimulator.Recording import Recording, RecordingType
import InternalSimulator.NetworkTransformation as NT



class NIR2FPGA:
    def __init__(
            self,
            name: str,
            nir_graph: nir.NIRGraph,
            dc: DiscretizationChoices,
            simulation_options: Optional[SimulationOptions] = None,
    ):
        """
        Initialize the NIR to FPGA exporter.

        Args:
            nir_graph: The NIR graph representation of the network
        """
        self.base_dir = Path(__file__).parent.parent.parent.parent

        if not type(dc).__name__ == "DiscretizationChoices":
            raise ValueError(
                "Parameter 'dc' must be an object of DiscretizationChoices"
            )

        self.name = name
        self.filename = re.sub(
            r'[<>:"/\\|?*\x00-\x1F]', "", re.sub(r"\s+", "_", self.name.strip())
        )[:255]
        self.timestamp = int(time.time())

        self.dc = dc

        self.timesteps = self.dc.timesteps

        self.dt = self.dc.dt_ms

        self.nir_graph: nir.NIRGraph = self._normalize_nir_graph(nir_graph, self.timesteps)
        self.transformation = NT.NetworkTransformation(
            nir_graph=self.nir_graph,
            dc=self.dc
        )
        self.data_sample = self.transformation.data_sample

        # Pull results from transformation stages
        internal_model = self.transformation.stages[RecordingType.INTERNAL].model
        assert internal_model is not None
        self.internal_model: torch.nn.Sequential = internal_model  # type: ignore[assignment]
        self.internal_ids = self.transformation.internal_ids
        self.quantized_model = self.transformation.stages[RecordingType.QUANTIZED].model
        quantized_ps = self.transformation.stages[RecordingType.QUANTIZED].parameter_space
        self.quantization_data: Dict[str, Any] = quantized_ps.properties  # type: ignore[attr-defined]
        self.rec_dict: Dict[str, Recording] = {
            "internal": self.transformation.stages[RecordingType.INTERNAL].recordings,
            "quantized": self.transformation.stages[RecordingType.QUANTIZED].recordings,
        }
        self.export_rec_dict: Dict[str, Recording] = dict(self.rec_dict)

        # Generate AXI packets from data_sample using IOManager
        # This creates pre-encoded packets with proper quantization
        # Extract input and output shapes from NIR graph (without timesteps)
        output_dims = self.nir_graph.nodes[list(self.nir_graph.output_type.keys())[0]].output_type["output"].tolist()  # type: ignore[attr-defined]

        self.io_manager = IOManager(
            input_shape=self.data_sample.squeeze(0).shape,
            output_shape=output_dims,
            quant_data=self.quantization_data
        )

        benchmark_metadata = {
            **self.dc.metadata(),
            "calibration_dataset_size": self.transformation.calibration_dataset_size,
        }

        self.save_dict = {
            "input": self.data_sample,
            "output": {"shape": output_dims},
            "quantizations": self.quantization_data,
            "timesteps": self.timesteps,
            "timestamp": self.timestamp,
            "benchmark": benchmark_metadata,
        }

        self.quantization_save_dict = {
            "quantizations": self.quantization_data,
            "timesteps": self.timesteps,
            "benchmark": benchmark_metadata,
        }

        self.files_dir: Optional[Path] = None
        self.nir_graph_trained: Optional[nir.NIRGraph] = None
        self.simulation = HardwareSimulation(self, simulation_options or SimulationOptions())
        self._plotter: Optional[Plotter] = None

        if self.dc.reduction:
            reduced_quantized = self._build_reduction_aware_quantized_recordings()
            self.export_rec_dict["quantized"] = Recording(RecordingType.QUANTIZED, reduced_quantized)

    @staticmethod
    def _normalize_nir_graph(nir_graph: nir.NIRGraph, timesteps: int) -> nir.NIRGraph:
        """Enforce the three normalization invariants and return a normalized copy.

        a. The graph is a strict linear chain (no branches, no cycles).
        b. Internal nodes are renamed to sequential integers "0", "1", … in chain order.
        c. No shape encodes the number of timesteps: if a node's input_type or
           output_type has a leading dimension equal to `timesteps`, that dimension
           is stripped. Timesteps are handled externally by the AXI packet stream.
        """
        node_ids = set(nir_graph.nodes.keys())
        if "input" not in node_ids or "output" not in node_ids:
            raise ValueError("NIR graph must have 'input' and 'output' nodes.")

        outgoing: Dict[str, list[str]] = {nid: [] for nid in node_ids}
        incoming: Dict[str, list[str]] = {nid: [] for nid in node_ids}
        for src, dst in nir_graph.edges:
            outgoing[src].append(dst)
            incoming[dst].append(src)

        if incoming["input"] or len(outgoing["input"]) != 1:
            raise ValueError("Graph is not a chain: 'input' must have exactly one successor and no predecessors.")
        if outgoing["output"] or len(incoming["output"]) != 1:
            raise ValueError("Graph is not a chain: 'output' must have exactly one predecessor and no successors.")
        for nid in node_ids - {"input", "output"}:
            if len(incoming[nid]) != 1 or len(outgoing[nid]) != 1:
                raise ValueError(
                    f"Graph is not a chain: node {nid!r} has {len(incoming[nid])} "
                    f"predecessor(s) and {len(outgoing[nid])} successor(s)."
                )

        ordered_internal: list[str] = []
        visited: set[str] = set()
        current = outgoing["input"][0]
        while current != "output":
            if current in visited:
                raise ValueError(f"Graph contains a cycle involving node {current!r}.")
            visited.add(current)
            ordered_internal.append(current)
            current = outgoing[current][0]

        if len(ordered_internal) != len(node_ids) - 2:
            raise ValueError("Graph is not a chain: not all internal nodes are reachable from 'input'.")

        name_map: Dict[str, str] = {old: str(i) for i, old in enumerate(ordered_internal)}
        name_map["input"] = "input"
        name_map["output"] = "output"

        new_nodes = {}
        for old_name, node in nir_graph.nodes.items():
            typed_node = cast(Any, copy.deepcopy(node))
            if hasattr(typed_node, "input_type"):
                for key, value in typed_node.input_type.items():
                    arr = np.array(value)
                    if len(arr) > 1 and int(arr[0]) == timesteps:
                        typed_node.input_type[key] = arr[1:]
            if hasattr(typed_node, "output_type"):
                for key, value in typed_node.output_type.items():
                    arr = np.array(value)
                    if len(arr) > 1 and int(arr[0]) == timesteps:
                        typed_node.output_type[key] = arr[1:]
            new_nodes[name_map[old_name]] = typed_node
        new_edges = [(name_map[src], name_map[dst]) for src, dst in nir_graph.edges]
        return nir.NIRGraph(nodes=new_nodes, edges=new_edges)

    @property
    def export_nir_graph(self) -> nir.NIRGraph:
        """The NIR graph used for export: the trained graph g' if set, else g.

        Equal to ``self.nir_graph`` until ``set_quantized_network`` is called,
        after which it is the QAT-trained graph held in ``self.nir_graph_trained``.
        """
        return self.nir_graph_trained if self.nir_graph_trained is not None else self.nir_graph

    def build_quantized_model(self, use_ste: bool = False) -> torch.nn.Sequential:
        # Structure comes from the calibrated QuantizationConfig.wrapper (built
        # from the NIR node with the correct spike_fn/norm_input/num_timesteps);
        # only QuantizationWrapper accepts those. Linear weights/biases are then
        # synced from internal_model so QAT-updated weights are picked up.
        quantized_layers: list[torch.nn.Module] = []
        for layer_id, layer in zip(self.internal_ids, self.internal_model):
            quant_cfg = self.quantization_data[layer_id]
            wrapper = copy.deepcopy(quant_cfg.wrapper)
            wrapper.use_ste = use_ste
            if wrapper.node_type in ("Linear", "Affine"):
                source = cast(Any, layer)
                with torch.no_grad():
                    wrapper.weight.data.copy_(source.weight.data)
                    if wrapper.bias is not None and getattr(source, "bias", None) is not None:
                        wrapper.bias.data.copy_(source.bias.data)
            else:
                wrapper.v_mem = None
                wrapper.i_syn = None
                # QAT backprops through the spike function, which needs a
                # surrogate gradient; the hard (use_ste=False) path does not.
                if use_ste and wrapper.spike_fn is not None:
                    wrapper.surrogate_grad_fn = sinabs.activation.SingleExponential()  # type: ignore[attr-defined]
            quantized_layers.append(wrapper)
        return torch.nn.Sequential(*quantized_layers)

    def get_quantized_network(self) -> torch.nn.Sequential:
        """Return the quantized network as a differentiable, learnable torch model.

        The result is a ``torch.nn.Sequential`` of ``QuantizationWrapper`` layers
        built with straight-through estimators, so gradients flow to the weight and
        bias ``Parameter``s of every Linear layer. Train or inspect it freely, then
        hand it back to ``set_quantized_network`` to export the trained weights.
        The network structure must not change between get and set.
        """
        return self.build_quantized_model(use_ste=True)

    def set_quantized_network(self, net: torch.nn.Sequential) -> None:
        """Accept a trained quantized network and route its weights to hardware.

        ``net`` must be structurally identical to what ``get_quantized_network``
        returned — same layers, shapes and neuron parameters — with only Linear
        weights/biases changed. The trained weights are written into a fresh graph
        g' (``self.nir_graph_trained``); the original graph g (``self.nir_graph``)
        is left untouched. All export recordings are regenerated so subsequent
        ``save_files``/``compile`` calls emit hardware for g'.

        Raises:
            ValueError: if ``net`` is not structurally identical to the reference.
        """
        reference = self.build_quantized_model(use_ste=True)
        self._assert_same_structure(net, reference)

        weights = self._extract_quantized_weights(net, self.internal_ids)

        # g stays immutable; g' is a fresh copy carrying the trained weights.
        trained_graph = copy.deepcopy(self.nir_graph)
        self._apply_weights_to_graph(trained_graph, weights)
        self.nir_graph_trained = trained_graph

        # Refresh the float internal model so recordings/output reflect training.
        self._apply_weights_to_internal_model(weights)
        self._refresh_export_recordings()

    def _refresh_export_recordings(self) -> None:
        """Rebuild internal/quantized recordings from the current (trained) weights."""
        internal_recordings = self.transformation._execute_network(
            self.internal_model,
            self.data_sample,
            self.transformation.single_save_hook,
        )
        hard_model = self.build_quantized_model(use_ste=False)
        quantized_recordings = self.transformation._execute_network(
            hard_model,
            self.data_sample,
            self.transformation.single_save_hook,
        )

        self.quantized_model = hard_model
        self.rec_dict["internal"] = Recording(RecordingType.INTERNAL, internal_recordings)
        self.rec_dict["quantized"] = Recording(RecordingType.QUANTIZED, quantized_recordings)
        self.export_rec_dict["internal"] = self.rec_dict["internal"]
        if self.dc.reduction:
            reduced_quantized = self._build_reduction_aware_quantized_recordings()
            self.export_rec_dict["quantized"] = Recording(RecordingType.QUANTIZED, reduced_quantized)
        else:
            self.export_rec_dict["quantized"] = self.rec_dict["quantized"]

    @staticmethod
    def _assert_same_structure(net: torch.nn.Sequential, reference: torch.nn.Sequential) -> None:
        """Raise ValueError unless `net` matches `reference` in everything but weights."""
        if len(net) != len(reference):
            raise ValueError(
                f"set_quantized_network: layer count mismatch — got {len(net)}, "
                f"expected {len(reference)}."
            )
        for idx, (raw_got, raw_ref) in enumerate(zip(net, reference)):
            if not isinstance(raw_got, QuantizationWrapper):
                raise ValueError(
                    f"set_quantized_network: layer {idx} is {type(raw_got).__name__}, "
                    f"expected QuantizationWrapper — pass the network from get_quantized_network()."
                )
            got = cast(Any, raw_got)
            ref = cast(Any, raw_ref)
            if got.node_type != ref.node_type:
                raise ValueError(
                    f"set_quantized_network: layer {idx} type mismatch — "
                    f"got {got.node_type!r}, expected {ref.node_type!r}."
                )
            if ref.node_type in ("Linear", "Affine"):
                if tuple(got.weight.shape) != tuple(ref.weight.shape):
                    raise ValueError(
                        f"set_quantized_network: layer {idx} weight shape mismatch — "
                        f"got {tuple(got.weight.shape)}, expected {tuple(ref.weight.shape)}."
                    )
                if (got.bias is None) != (ref.bias is None):
                    raise ValueError(f"set_quantized_network: layer {idx} bias presence changed.")
                if (
                    got.bias is not None
                    and ref.bias is not None
                    and tuple(got.bias.shape) != tuple(ref.bias.shape)
                ):
                    raise ValueError(
                        f"set_quantized_network: layer {idx} bias shape mismatch — "
                        f"got {tuple(got.bias.shape)}, expected {tuple(ref.bias.shape)}."
                    )
            else:
                NIR2FPGA._assert_same_neuron_params(idx, got, ref)

    @staticmethod
    def _assert_same_neuron_params(idx: int, got: Any, ref: Any) -> None:
        """Raise ValueError if any non-weight neuron parameter differs."""
        for attr in ("num_timesteps", "alpha_mem", "alpha_syn"):
            if getattr(got, attr) != getattr(ref, attr):
                raise ValueError(
                    f"set_quantized_network: layer {idx} {attr} changed; only weights may change."
                )
        if not torch.equal(got.spike_threshold, ref.spike_threshold):
            raise ValueError(
                f"set_quantized_network: layer {idx} spike_threshold changed; only weights may change."
            )
        got_min, ref_min = got.min_v_mem, ref.min_v_mem
        if (got_min is None) != (ref_min is None) or (
            got_min is not None and ref_min is not None and not torch.equal(got_min, ref_min)
        ):
            raise ValueError(
                f"set_quantized_network: layer {idx} min_v_mem changed; only weights may change."
            )

    @staticmethod
    def _extract_quantized_weights(
        net: torch.nn.Sequential,
        layer_ids: List[str],
    ) -> Dict[str, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Pull weight/bias tensors from the Linear QuantizationWrapper layers."""
        weights: Dict[str, tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        for layer_id, layer in zip(layer_ids, net):
            wrapper = cast(Any, layer)
            if getattr(wrapper, "node_type", None) not in ("Linear", "Affine"):
                continue
            weight = wrapper.weight.detach().cpu()
            bias = wrapper.bias.detach().cpu() if wrapper.bias is not None else None
            weights[layer_id] = (weight, bias)
        return weights

    def _apply_weights_to_internal_model(
        self,
        weights: Dict[str, tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> None:
        for layer_id, layer in zip(self.internal_ids, self.internal_model):
            if layer_id not in weights:
                continue
            weight, bias = weights[layer_id]
            target = cast(Any, layer)
            with torch.no_grad():
                target.weight.data.copy_(weight)
                if bias is not None and getattr(target, "bias", None) is not None:
                    target.bias.data.copy_(bias)

    @staticmethod
    def _apply_weights_to_graph(
        graph: nir.NIRGraph,
        weights: Dict[str, tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> None:
        for layer_id, (weight, bias) in weights.items():
            if layer_id not in graph.nodes:
                continue
            node = cast(Any, graph.nodes[layer_id])
            if hasattr(node, "weight"):
                node.weight = np.asarray(weight.tolist(), dtype=np.float32)
            if bias is not None and getattr(node, "bias", None) is not None:
                node.bias = np.asarray(bias.tolist(), dtype=np.float32)

    @staticmethod
    def _tensor_2d(value: Any) -> torch.Tensor:
        tensor = value.clone().detach() if isinstance(value, torch.Tensor) else torch.tensor(value)
        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for export recordings, got shape {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D recording tensor, got shape {tuple(tensor.shape)}")
        return tensor.float()

    @staticmethod
    def _node_scalar(node: Any, field: str) -> float:
        value = getattr(node, field)
        return float(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def _full_range_quant(bits: int, frac_bits: int, signed: bool) -> Dict[str, Any]:
        if signed:
            min_value = -(1 << (bits - 1))
            max_value = (1 << (bits - 1)) - 1
        else:
            min_value = 0
            max_value = (1 << bits) - 1
        return {
            "bits": bits,
            "frac_bits": frac_bits,
            "int_bits": bits - frac_bits,
            "signed": signed,
            "exp": -frac_bits,
            "min_value": min_value,
            "max_value": max_value,
        }

    @staticmethod
    def _widen_v_mem_quant(quant: Dict[str, Any]) -> Dict[str, Any]:
        extra_bits = quant["bits"] // 2
        extra_frac = quant["frac_bits"] // 2
        return NIR2FPGA._full_range_quant(
            quant["bits"] + extra_bits,
            quant["frac_bits"] + extra_frac,
            quant["signed"],
        )

    @staticmethod
    def _quantize_to_int(x: torch.Tensor, quant: Dict[str, Any], trunc: bool = False) -> torch.Tensor:
        scale = 2 ** quant["exp"]
        scaled = x / scale
        raw = torch.trunc(scaled) if trunc else torch.floor(scaled)
        return NIR2FPGA._wrap_raw_tensor(raw.long(), quant["bits"], quant["signed"])

    @staticmethod
    def _wrap_raw_tensor(raw: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
        if signed:
            min_raw = -(1 << (bits - 1))
            max_raw = (1 << (bits - 1)) - 1
        else:
            min_raw = 0
            max_raw = (1 << bits) - 1
        wrapped = torch.clamp(raw.long(), min_raw, max_raw)
        return wrapped

    @staticmethod
    def _dequantize_from_int(raw: torch.Tensor, quant: Dict[str, Any]) -> torch.Tensor:
        return raw.float() * (2 ** quant["exp"])

    @staticmethod
    def _coeff_mul_to_raw(
        values_raw: torch.Tensor,
        coeff: float,
        in_quant: Dict[str, Any],
        out_quant: Dict[str, Any],
        extra_frac_bits: int = 8,
    ) -> torch.Tensor:
        coeff_frac = out_quant["frac_bits"] + extra_frac_bits
        coeff_raw = math.floor(coeff * (2 ** coeff_frac))
        shift = coeff_frac + in_quant["frac_bits"] - out_quant["frac_bits"]
        prod = torch.floor(values_raw.float() * coeff_raw / (2.0 ** shift))
        return NIR2FPGA._wrap_raw_tensor(prod.long(), out_quant["bits"], out_quant["signed"])

    def _simulate_reduced_affine_lif(
        self,
        linear_id: str,
        lif_id: str,
        input_dense: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
        linear_node = cast(Any, self.export_nir_graph.nodes[linear_id])
        lif_node = cast(Any, self.export_nir_graph.nodes[lif_id])

        input_q = self.quantization_data[linear_id]["input"]
        weight_q = self.quantization_data[linear_id]["weights"]
        linear_out_q = self.quantization_data[linear_id]["output"]
        spike_q = self.quantization_data[lif_id]["output"]
        v_mem_q = self._widen_v_mem_quant(self.quantization_data[lif_id]["v_mem"])

        input_raw = self._quantize_to_int(input_dense, input_q)
        weights = torch.tensor(np.asarray(linear_node.weight), dtype=torch.float32)
        weight_raw = self._quantize_to_int(weights, weight_q, trunc=True)

        alpha_mem = float(np.exp(-1.0 / self._node_scalar(lif_node, "tau")))
        input_factor = self._node_scalar(lif_node, "r") * (1.0 - alpha_mem)
        threshold_raw = self._quantize_to_int(
            torch.tensor([self._node_scalar(lif_node, "v_threshold")], dtype=torch.float32),
            v_mem_q,
        )[0]

        exp_shift = input_q["exp"] + weight_q["exp"] - linear_out_q["exp"]
        timesteps, _ = input_raw.shape
        out_features = weight_raw.shape[0]
        state_raw = torch.zeros(out_features, dtype=torch.long)
        outputs = torch.zeros((timesteps, out_features), dtype=torch.float32)
        v_mem = torch.zeros((timesteps, out_features), dtype=torch.float32)
        output_events: list[list[tuple[int, int]]] = []

        for timestep in range(timesteps):
            events = torch.nonzero(input_raw[timestep], as_tuple=False).flatten().tolist()
            if not events:
                events = [0]
            spikes_this_timestep: list[tuple[int, int]] = []

            for event_idx, input_coord in enumerate(events):
                input_value_raw = int(input_raw[timestep, input_coord].item())
                products = torch.floor(
                    input_value_raw * weight_raw[:, input_coord].float() * (2.0 ** exp_shift)
                )
                products_raw = self._wrap_raw_tensor(products.long(), linear_out_q["bits"], linear_out_q["signed"])

                leak_coeff = 1.0 if (timestep == 0 or event_idx > 0) else alpha_mem
                leaked_raw = self._coeff_mul_to_raw(state_raw, leak_coeff, v_mem_q, v_mem_q)
                input_contrib_raw = self._coeff_mul_to_raw(products_raw, input_factor, linear_out_q, v_mem_q)
                updated_raw = self._wrap_raw_tensor(
                    leaked_raw + input_contrib_raw,
                    v_mem_q["bits"],
                    v_mem_q["signed"],
                )
                will_spike = updated_raw >= threshold_raw
                state_raw = torch.where(will_spike, torch.zeros_like(updated_raw), updated_raw)
                outputs[timestep] = outputs[timestep] + self._dequantize_from_int(will_spike.long(), spike_q)
                spike_indices = torch.nonzero(will_spike, as_tuple=False).flatten().tolist()
                spikes_this_timestep.extend((idx, 1) for idx in spike_indices)

            v_mem[timestep] = self._dequantize_from_int(state_raw, v_mem_q)
            output_events.append(spikes_this_timestep)

        return outputs, v_mem, output_events

    def _simulate_affine(
        self,
        layer_id: str,
        input_dense: torch.Tensor,
        input_events: Optional[list[list[tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        node = cast(Any, self.export_nir_graph.nodes[layer_id])
        input_q = self.quantization_data[layer_id]["input"]
        weight_q = self.quantization_data[layer_id]["weights"]
        output_q = self.quantization_data[layer_id]["output"]

        input_raw = self._quantize_to_int(input_dense, input_q)
        weights = torch.tensor(np.asarray(node.weight), dtype=torch.float32)
        weight_raw = self._quantize_to_int(weights, weight_q, trunc=True)
        exp_shift = input_q["exp"] + weight_q["exp"] - output_q["exp"]
        accum_bits = output_q["bits"] + int(np.ceil(np.log2(max(1, weight_raw.shape[1]))))

        outputs_raw = torch.zeros((input_raw.shape[0], weight_raw.shape[0]), dtype=torch.long)
        for timestep in range(input_raw.shape[0]):
            accum = torch.zeros(weight_raw.shape[0], dtype=torch.long)
            timestep_events = input_events[timestep] if input_events is not None else [
                (coord, int(input_raw[timestep, coord].item()))
                for coord in torch.nonzero(input_raw[timestep], as_tuple=False).flatten().tolist()
            ]
            for input_coord, input_value_raw in timestep_events:
                products = torch.floor(
                    input_value_raw * weight_raw[:, input_coord].float() * (2.0 ** exp_shift)
                )
                accum = self._wrap_raw_tensor(
                    accum + products.long(),
                    accum_bits,
                    output_q["signed"],
                )

            if hasattr(node, "bias") and getattr(node, "bias") is not None:
                bias = torch.tensor(np.asarray(node.bias), dtype=torch.float32)
                accum = self._wrap_raw_tensor(
                    accum + self._quantize_to_int(bias, output_q),
                    accum_bits,
                    output_q["signed"],
                )

            outputs_raw[timestep] = self._wrap_raw_tensor(accum, output_q["bits"], output_q["signed"])

        return self._dequantize_from_int(outputs_raw, output_q)

    def _simulate_li(self, layer_id: str, input_dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node = cast(Any, self.export_nir_graph.nodes[layer_id])
        input_q = self.quantization_data[layer_id]["input"]
        v_mem_q = self.quantization_data[layer_id]["v_mem"]
        output_q = self.quantization_data[layer_id]["output"]

        input_raw = self._quantize_to_int(input_dense, input_q)
        alpha_mem = float(np.exp(-1.0 / self._node_scalar(node, "tau")))
        input_factor = self._node_scalar(node, "r") * alpha_mem
        state_raw = torch.zeros(input_raw.shape[1], dtype=torch.long)
        outputs_raw = torch.zeros_like(input_raw)
        v_mem_raw = torch.zeros_like(input_raw)

        for timestep in range(input_raw.shape[0]):
            leaked_raw = self._coeff_mul_to_raw(state_raw, alpha_mem, v_mem_q, v_mem_q)
            input_contrib_raw = self._coeff_mul_to_raw(input_raw[timestep], input_factor, input_q, v_mem_q)
            state_raw = self._wrap_raw_tensor(
                leaked_raw + input_contrib_raw,
                v_mem_q["bits"],
                v_mem_q["signed"],
            )
            v_mem_raw[timestep] = state_raw
            outputs_raw[timestep] = self._quantize_to_int(self._dequantize_from_int(state_raw, v_mem_q), output_q)

        return self._dequantize_from_int(outputs_raw, output_q), self._dequantize_from_int(v_mem_raw, v_mem_q)

    def _simulate_lif(self, layer_id: str, input_dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node = cast(Any, self.export_nir_graph.nodes[layer_id])
        input_q = self.quantization_data[layer_id]["input"]
        v_mem_q = self.quantization_data[layer_id]["v_mem"]
        output_q = self.quantization_data[layer_id]["output"]

        input_raw = self._quantize_to_int(input_dense, input_q)
        alpha_mem = float(np.exp(-1.0 / self._node_scalar(node, "tau")))
        input_factor = self._node_scalar(node, "r") * (1.0 - alpha_mem)
        threshold_raw = self._quantize_to_int(
            torch.tensor([self._node_scalar(node, "v_threshold")], dtype=torch.float32),
            v_mem_q,
        )[0]

        state_raw = torch.zeros(input_raw.shape[1], dtype=torch.long)
        outputs_raw = torch.zeros_like(input_raw)
        v_mem_raw = torch.zeros_like(input_raw)

        for timestep in range(input_raw.shape[0]):
            leaked_raw = self._coeff_mul_to_raw(state_raw, alpha_mem, v_mem_q, v_mem_q)
            input_contrib_raw = self._coeff_mul_to_raw(input_raw[timestep], input_factor, input_q, v_mem_q)
            updated_raw = self._wrap_raw_tensor(
                leaked_raw + input_contrib_raw,
                v_mem_q["bits"],
                v_mem_q["signed"],
            )
            will_spike = updated_raw >= threshold_raw
            state_raw = torch.where(will_spike, torch.zeros_like(updated_raw), updated_raw)
            outputs_raw[timestep] = self._wrap_raw_tensor(will_spike.long(), output_q["bits"], output_q["signed"])
            v_mem_raw[timestep] = state_raw

        return self._dequantize_from_int(outputs_raw, output_q), self._dequantize_from_int(v_mem_raw, v_mem_q)

    def _build_reduction_aware_quantized_recordings(self) -> Dict[str, Dict[str, torch.Tensor]]:
        base_recordings = {
            layer_id: {name: value.clone().detach() for name, value in layer_rec.items()}
            for layer_id, layer_rec in self.rec_dict["quantized"].data.items()
        }

        current_dense: Optional[torch.Tensor] = None
        current_events: Optional[list[list[tuple[int, int]]]] = None
        idx = 0
        while idx < len(self.internal_ids):
            layer_id = self.internal_ids[idx]
            node = cast(Any, self.export_nir_graph.nodes[layer_id])
            kind = type(node).__name__

            if current_dense is None and "input" in base_recordings.get(layer_id, {}):
                current_dense = self._tensor_2d(base_recordings[layer_id]["input"])

            if kind == "Affine" and idx + 1 < len(self.internal_ids):
                next_id = self.internal_ids[idx + 1]
                next_node = cast(Any, self.export_nir_graph.nodes[next_id])
                if type(next_node).__name__ == "LIF":
                    assert current_dense is not None
                    base_recordings[layer_id]["input"] = current_dense.clone()
                    fused_output, fused_v_mem, fused_events = self._simulate_reduced_affine_lif(layer_id, next_id, current_dense)
                    base_recordings[next_id]["input"] = current_dense.clone()
                    base_recordings[next_id]["output"] = fused_output.clone()
                    base_recordings[next_id]["v_mem"] = fused_v_mem.clone()
                    if "i_syn" in base_recordings[next_id]:
                        del base_recordings[next_id]["i_syn"]
                    current_dense = fused_output
                    current_events = fused_events
                    idx += 2
                    continue

            if kind == "Affine":
                assert current_dense is not None
                base_recordings[layer_id]["input"] = current_dense.clone()
                next_dense = self._simulate_affine(layer_id, current_dense, current_events)
                base_recordings[layer_id]["output"] = next_dense.clone()
                current_dense = next_dense
                current_events = None
            elif kind == "LI":
                assert current_dense is not None
                base_recordings[layer_id]["input"] = current_dense.clone()
                li_output, li_v_mem = self._simulate_li(layer_id, current_dense)
                base_recordings[layer_id]["output"] = li_output.clone()
                base_recordings[layer_id]["v_mem"] = li_v_mem.clone()
                current_dense = li_output
                current_events = None
            elif kind == "LIF":
                if current_dense is None:
                    current_dense = self._tensor_2d(base_recordings[layer_id]["output"])
                current_events = None
            else:
                if layer_id in base_recordings and "output" in base_recordings[layer_id]:
                    current_dense = self._tensor_2d(base_recordings[layer_id]["output"])
                current_events = None
            idx += 1

        if current_dense is not None and "output" in base_recordings and "input" in base_recordings["output"]:
            base_recordings["output"]["input"] = current_dense.clone()

        return base_recordings

    def build_hardware_reference_quantized_recordings(self) -> Dict[str, Dict[str, torch.Tensor]]:
        base_recordings = {
            layer_id: {name: value.clone().detach() for name, value in layer_rec.items()}
            for layer_id, layer_rec in self.rec_dict["quantized"].data.items()
        }

        current_dense: Optional[torch.Tensor] = None
        current_events: Optional[list[list[tuple[int, int]]]] = None
        idx = 0
        while idx < len(self.internal_ids):
            layer_id = self.internal_ids[idx]
            node = cast(Any, self.export_nir_graph.nodes[layer_id])
            kind = type(node).__name__

            if current_dense is None and "input" in base_recordings.get(layer_id, {}):
                current_dense = self._tensor_2d(base_recordings[layer_id]["input"])

            if kind == "Affine":
                assert current_dense is not None
                base_recordings[layer_id]["input"] = current_dense.clone()
                next_dense = self._simulate_affine(layer_id, current_dense, current_events)
                base_recordings[layer_id]["output"] = next_dense.clone()
                current_dense = next_dense
                current_events = None
            elif kind == "LI":
                assert current_dense is not None
                base_recordings[layer_id]["input"] = current_dense.clone()
                li_output, li_v_mem = self._simulate_li(layer_id, current_dense)
                base_recordings[layer_id]["output"] = li_output.clone()
                base_recordings[layer_id]["v_mem"] = li_v_mem.clone()
                current_dense = li_output
                current_events = None
            elif kind == "LIF":
                assert current_dense is not None
                base_recordings[layer_id]["input"] = current_dense.clone()
                lif_output, lif_v_mem = self._simulate_lif(layer_id, current_dense)
                base_recordings[layer_id]["output"] = lif_output.clone()
                base_recordings[layer_id]["v_mem"] = lif_v_mem.clone()
                if "i_syn" in base_recordings[layer_id]:
                    del base_recordings[layer_id]["i_syn"]
                current_dense = lif_output
                current_events = None
            else:
                if layer_id in base_recordings and "output" in base_recordings[layer_id]:
                    current_dense = self._tensor_2d(base_recordings[layer_id]["output"])
                current_events = None
            idx += 1

        if current_dense is not None and "output" in base_recordings and "input" in base_recordings["output"]:
            base_recordings["output"]["input"] = current_dense.clone()

        return base_recordings

    def sync_nir_graph_from_internal_model(self) -> None:
        for layer_id, layer in zip(self.internal_ids, self.internal_model):
            if type(layer).__name__ != "Linear":
                continue
            if layer_id not in self.nir_graph.nodes:
                continue
            node = cast(Any, self.nir_graph.nodes[layer_id])
            if hasattr(node, "weight"):
                weight = cast(Any, layer.weight)
                node.weight = np.asarray(weight.detach().cpu().tolist(), dtype=np.float32)
            if hasattr(node, "bias") and getattr(layer, "bias", None) is not None:
                bias = cast(Any, layer.bias)
                node.bias = np.asarray(bias.detach().cpu().tolist(), dtype=np.float32)

    def add_recording(self,
                      rec_type: str,
                      recording: Dict[str, Dict[str, torch.Tensor]],
                      observed_accuracy: Optional[float] = None) -> None:
        """
        Add a recording to the rec_dict for later plotting/saving.

        Args:
            rec_type: One of "source", "behavioural", or "hardware"
            recording: Dict with layer_id keys, each containing signal tensors.
            observed_accuracy: Known accuracy for SOURCE recordings. When provided,
                evaluate() will report this value directly instead of recomputing it.
                Ignored (with a warning) for non-SOURCE recording types.
        """
        type_map = {
            "source": RecordingType.SOURCE,
            "behavioural": RecordingType.BEHAVIOURAL,
            "hardware": RecordingType.HARDWARE,
        }
        if rec_type not in type_map:
            raise ValueError(f"rec_type must be one of {list(type_map.keys())}, got '{rec_type}'")
        self.transformation.add_recording(type_map[rec_type], recording, observed_accuracy=observed_accuracy)
        self.rec_dict[rec_type] = self.transformation.stages[type_map[rec_type]].recordings

    def compile(self) -> None:
        """Compile the Scala backend without running a full simulation."""
        self.simulation.compile()

    def simulate(self,
                 sample_index: int
                 ) -> None:
        """Run hardware simulation via the Scala backend."""
        sample_data, _ = self.dc.dataset[sample_index]
        sample_batch = sample_data.unsqueeze(0).float()  # (1, T, N)


        assert isinstance(self.quantized_model, torch.nn.Sequential)
        quant_recs = self.transformation._execute_network(
            self.quantized_model,
            sample_batch,
            NT.NetworkTransformation.single_save_hook
        )
        output_recording = quant_recs["output"]["input"]

        self.simulation.simulate(sample=sample_data,
                                 output_recording = output_recording,
                                 skip_vcd=True)


    def _extract_recording(
        self,
        rec_type: str,
        layer_id: str,
        index: int,
        ignorelist: Optional[List[str]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:  # type: ignore[type-arg]
        """
        Extract a 1D (T,) recording for a given layer and neuron index.

        For hardware: pulls from simulation.get_recording() (already 1D).
        For all other types: slices arr[:, index] from rec_dict tensors.
        Returns None if the recording type is unavailable or ignored.
        """
        if ignorelist and rec_type in ignorelist:
            return None

        if rec_type == "hardware":
            if self.simulation._hw_vars is None:
                return None
            return self.simulation.get_recording(layer_id, index)

        if rec_type not in self.rec_dict:
            return None
        rec = self.rec_dict[rec_type]
        data = rec.data
        if layer_id not in data:
            return None

        layer_data = data[layer_id]
        has_v_mem = "v_mem" in layer_data
        has_output = "output" in layer_data
        if not has_v_mem and not has_output:
            return None

        def _to_1d(val: Any) -> np.ndarray:  # type: ignore[type-arg]
            if isinstance(val, torch.Tensor):
                arr = val.detach().numpy()
            else:
                arr = np.array(val)
            if arr.ndim == 3:
                arr = arr[0]
            return arr[:, index]

        result: Dict[str, np.ndarray] = {}  # type: ignore[type-arg]
        if has_v_mem:
            result["v_mem"] = _to_1d(layer_data["v_mem"])
        if has_output:
            result["output"] = _to_1d(layer_data["output"])
        return result

    def plot(
        self,
        layer_id: str,
        index: int,
        filename: Optional[str] = None,
        ignorelist: Optional[List[str]] = None,
        plot_size: tuple[int, int] = (15, 15),
    ) -> None:
        """
        Plot recordings for a specific layer and neuron index.

        Args:
            layer_id: The layer ID to plot (matches normalised NIR graph node ID).
            index: Which neuron/output to plot.
            filename: Save path (if None, shows plot).
            ignorelist: Recording names to skip. Options: "source", "internal",
                        "quantized", "behavioural", "hardware".
            plot_size: Figure size (width, height).
        """
        recordings: Dict[str, Dict[str, np.ndarray]] = {}  # type: ignore[type-arg]
        for rec_type in ("source", "internal", "quantized", "behavioural", "hardware"):
            rec = self._extract_recording(rec_type, layer_id, index, ignorelist)
            if rec is not None:
                recordings[rec_type] = rec

        plotter = Plotter(self.nir_graph) if self._plotter is None else self._plotter
        self._plotter = plotter
        plotter.plot(layer_id, recordings, filename, plot_size)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_accuracy(self, prediction_fn: Any) -> None:
        """
        Print an accuracy table for all software stages.

        Errors if a HARDWARE stage is present — running sbt per batch is too slow.
        Delegates to NetworkTransformation.evaluate() for SOURCE/INTERNAL/QUANTIZED/BEHAVIOURAL.
        """
        if RecordingType.HARDWARE in self.transformation.stages:
            raise NotImplementedError(
                "Hardware accuracy evaluation is not supported — "
                "running the full sbt simulation per batch would take too long."
            )
        self.transformation.evaluate(prediction_fn)

    def evaluate_characteristic(
        self,
        sample_index: int,
        output_dir: Optional[Path] = None,
        plot_size: tuple[int, int] = (20, 10),
        r_threshold: float = 0.95,
    ) -> None:
        """
        For a single input sample, plot and score every NIR node × neuron index.

        Runs INTERNAL and QUANTIZED fresh on sample_index. Uses rec_dict for
        SOURCE/BEHAVIOURAL. Uses simulation.get_recording() for HARDWARE.
        Computes Pearson r (vs INTERNAL) for each (stage, node, param) and prints
        a summary table; PASS when mean r ≥ r_threshold (default 0.95).
        """
        sample_data, _ = self.dc.dataset[sample_index]
        sample_batch = sample_data.unsqueeze(0).float()  # (1, T, N)

        # Run each model-backed stage on this specific sample
        stage_recs: Dict[str, Any] = {}
        for key, rt in (("internal", RecordingType.INTERNAL), ("quantized", RecordingType.QUANTIZED)):
            stage = self.transformation.stages[rt]
            assert stage.model is not None
            stage_recs[key] = self.transformation._execute_network(
                cast(torch.nn.Sequential, stage.model), sample_batch, NT.NetworkTransformation.single_save_hook
            )
        # Save for Spinal simulation's outputCheck
        output_recording = stage_recs["quantized"]["output"]["input"][0]

        # Hardware: require a successful compile with unchanged primitives
        if not self.simulation._compile_successful:
            raise RuntimeError(
                "[NIR2FPGA] Hardware simulation requires a successful compile. Call compile() first."
            )
        current_hash = self.simulation._hash_primitives_dir()
        if current_hash != self.simulation._compile_hash:
            raise RuntimeError(
                "[NIR2FPGA] Primitives directory has changed since the last compile. Call compile() again."
            )
        self.simulation.simulate(sample=sample_data, output_recording=output_recording)

        plotter = self._plotter if self._plotter is not None else Plotter(self.nir_graph)
        self._plotter = plotter

        _PLOTTABLE = (nir.LIF, nir.IF, nir.CubaLIF, nir.LI, nir.I, nir.Affine, nir.Linear)

        # results[stage][node_id][param] = list of r per neuron index
        results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        # raw_arrays[stage][node_id][param] = list of (T,) arrays, one per neuron index
        raw_arrays: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {}

        for node_id, node in self.nir_graph.nodes.items():
            if isinstance(node, (nir.Input, nir.Output)):
                continue
            if not isinstance(node, _PLOTTABLE):
                continue

            neuron_count = self._get_neuron_count(node)
            hw_probe_limit = (
                min(neuron_count, 10) if isinstance(node, (nir.LIF, nir.IF, nir.CubaLIF))
                else neuron_count
            )

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)

            for index in range(neuron_count):
                recordings: Dict[str, Dict[str, np.ndarray]] = {}

                for key in ("internal", "quantized"):
                    rec = self._stage_recs_to_1d(stage_recs[key], node_id, index)
                    if rec is not None:
                        recordings[key] = rec

                for key in ("source", "behavioural"):
                    rec = self._extract_recording(key, node_id, index)
                    if rec is not None:
                        recordings[key] = rec

                if self.simulation._hw_vars is not None:
                    hw_params: Optional[List[str]] = None if index < hw_probe_limit else ["output"]
                    try:
                        recordings["hardware"] = self.simulation.get_recording(node_id, index, hw_params)
                    except (KeyError, IndexError):
                        pass

                if not recordings:
                    continue

                for stage_key, stage_rec in recordings.items():
                    for param, arr in stage_rec.items():
                        (
                            raw_arrays
                            .setdefault(stage_key, {})
                            .setdefault(node_id, {})
                            .setdefault(param, [])
                            .append(arr)
                        )

                _PIPELINE = ["source", "internal", "quantized", "behavioural", "hardware"]
                available = [s for s in _PIPELINE if s in recordings]
                for i in range(1, len(available)):
                    prev_key = available[i - 1]
                    stage_key = available[i]
                    ref = recordings[prev_key]
                    stage_rec = recordings[stage_key]
                    for param, arr in stage_rec.items():
                        if param not in ref:
                            continue
                        r = self._pearson_r(ref[param], arr)
                        (
                            results
                            .setdefault(stage_key, {})
                            .setdefault(node_id, {})
                            .setdefault(param, [])
                            .append(r)
                        )

                filename = str(output_dir / f"{node_id}_{index}.png") if output_dir else None
                plotter.plot(node_id, recordings, filename, plot_size)

        if output_dir is not None and raw_arrays:
            npz_data: Dict[str, np.ndarray] = {"sample_index": np.array(sample_index)}
            for stage, node_dict in raw_arrays.items():
                for nid, param_dict in node_dict.items():
                    for param, arrays in param_dict.items():
                        npz_data[f"{stage}__{nid}__{param}"] = np.stack(arrays)  # (N, T)
            np.savez(str(output_dir / "characteristic.npz"), **npz_data)

        _DISPLAY_NAME = {
            "source": "Source",
            "quantized": "Quantization",
            "behavioural": "Behavioural",
            "hardware": "Compilation",
        }
        _ORDER = ["source", "quantized", "behavioural", "hardware"]

        print(f"\nCharacteristic Evaluation — sample {sample_index}")
        header = f"{'Transformation':<14} {'Node':>6} {'NIR Type':<10} {'Param':>8} {'r':>8}  Pass"
        print(header)
        print("-" * len(header))
        for stage_key in [k for k in _ORDER if k in results]:
            node_dict = results[stage_key]
            label = _DISPLAY_NAME.get(stage_key, stage_key)
            for nid, param_dict in sorted(node_dict.items()):
                nir_type = type(self.nir_graph.nodes[nid]).__name__ if nid in self.nir_graph.nodes else "?"
                for param, r_vals in sorted(param_dict.items()):
                    mean_r = float(np.mean(r_vals))
                    passed = "✓" if mean_r >= r_threshold else "✗"
                    print(f"{label:<14} {nid:>6} {nir_type:<10} {param:>8} {mean_r:>8.4f}  {passed}")

    def _pearson_r(self, a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[type-arg]
        """Pearson r guarded against zero-variance inputs. Returns 1.0 iff both constant and equal."""
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 1.0 if np.array_equal(a, b) else 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _get_neuron_count(self, node: Any) -> int:
        """Output dimension of a NIR node, derived from node attributes (not internal_model)."""
        for attr in ("v_threshold", "tau_mem"):
            if hasattr(node, attr):
                arr = np.asarray(getattr(node, attr))
                return int(arr.shape[0]) if arr.ndim > 0 else 1
        if hasattr(node, "weight"):
            return int(np.asarray(node.weight).shape[0])
        return 1

    def _stage_recs_to_1d(
        self,
        recs: Dict[str, Any],
        layer_id: str,
        index: int,
    ) -> Optional[Dict[str, np.ndarray]]:  # type: ignore[type-arg]
        """Slice neuron `index` from a {layer_id: {param: (T, N) tensor}} recordings dict."""
        if layer_id not in recs:
            return None
        result: Dict[str, np.ndarray] = {}  # type: ignore[type-arg]
        for key in ("v_mem", "output"):
            val = recs[layer_id].get(key)
            if val is None:
                continue
            arr = val.detach().numpy() if isinstance(val, torch.Tensor) else np.array(val)
            if arr.ndim >= 2:
                result[key] = arr[:, index]
        return result if result else None

    def __str__(self):
        lines = ["{"]
        for key, value in self.save_dict.items():
            line = f"    {key}: "
            if isinstance(value, torch.Tensor):
                line += str(value.shape)
            elif isinstance(value, dict):
                line += f"{value.keys()}"
            else:
                line += str(value)
            lines.append(line)
        lines.append("}")

        return "\n".join(lines)

    def _default_serializer(self, o: Any) -> Any:
        if isinstance(o, Recording):
            return o.data
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, QuantizationConfig):
            return o.quants
        raise TypeError(
            f"Object of type {type(o).__name__} is not JSON serializable"
        )

    def save_recordings(self, directory=Path("./")):
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        json_file = dir_path / f"{self.name}.json"

        with open(json_file, "w") as f:
            json.dump(self.rec_dict, f, default=self._default_serializer)
        print(f"Wrote {json_file}.")


    def save_files(self, directory=Path("./")):
        dir_path = Path(directory) / self.filename
        dir_path.mkdir(parents=True, exist_ok=True)
        nir_file = dir_path / "model.nir"
        json_file = dir_path / "model.json"
        quant_file = dir_path / "quantizations.json"
        compilation_file = dir_path / "compilation.json"
        input_npy = dir_path / "input.npy"
        output_npy = dir_path / "output.npy"
        recordings_npy = dir_path / "recordings.npy"

        with open(compilation_file, "w") as f:
            json.dump(
                {
                    "dataset_name": self.dc.dataset_name,
                    "macWidth": self.dc.macWidth,
                    "reduction": self.dc.reduction,
                    "spikeGating": True,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        with open(quant_file, "w") as f:
            json.dump(self.quantization_save_dict, f, default=self._default_serializer)

        export_dict = dict(self.save_dict)
        with open(json_file, "w") as f:
            json.dump(export_dict, f, default=self._default_serializer)

        packets_npy = dir_path / "input_packets.npy"
        np.save(str(packets_npy), self.io_manager.create_input_packets_numpy(
            self.data_sample.squeeze(0).detach().numpy()
        ))

        nir.write(nir_file, self.export_nir_graph)

        np.save(input_npy, self.data_sample.detach().numpy())
        with torch.no_grad():
            output = self.internal_model(self.data_sample.flatten(0, 1))
            output = output.view(self.data_sample.shape[0], self.data_sample.shape[1], *output.shape[1:])
        np.save(output_npy, output.detach().numpy())

        # Expected quantized output trace for hardware outputCheck. model.json no
        # longer carries recordings — this npy is the sole expected-output channel.
        quantized_output = self.export_rec_dict["quantized"]["output"]["input"]
        quantized_output = np.asarray(quantized_output.detach().cpu().numpy())
        if quantized_output.ndim == 3:
            quantized_output = quantized_output[0]  # drop leading batch dim → (T, N_out)
        np.save(recordings_npy, quantized_output)

        self.files_dir = dir_path.resolve()
        print(
            f"Wrote {nir_file}, {json_file}, {quant_file}, {compilation_file}, "
            f"{input_npy}, {output_npy}, {recordings_npy}, and {packets_npy}."
        )

    def save_throughput_packets(self, n: int = 100, directory: Path = Path("./")) -> None:
        """Encode up to n dataset samples as AXI packets and write throughput_packets.json."""
        dir_path = Path(directory) / self.filename
        dir_path.mkdir(parents=True, exist_ok=True)

        dataset = self.dc.dataset
        num_samples = min(n, len(cast(Sized, dataset)))

        all_packets: List[List[int]] = []
        for i in range(num_samples):
            sample, _ = dataset[i]  # shape: (timesteps, *input_dims)
            packet_list = self.io_manager.create_input_packets(sample)
            all_packets.append(self.io_manager.packets_to_numpy(packet_list).tolist())

        throughput_file = dir_path / "throughput_packets.json"
        with open(throughput_file, "w") as f:
            json.dump({"num_samples": num_samples, "timesteps": self.timesteps, "samples": all_packets}, f)
        print(f"Wrote {throughput_file} with {num_samples} samples.")

    def report_quantization(self):
        print(f'Quantization data of network "{self.name}"')
        for layer in self.internal_ids:
            print(f"{layer}:")
            for param in self.quantization_data[layer].keys():
                print(f"\t{param}: {self.quantization_data[layer].to_qformat(param)}")

    def verify_hardware(self) -> bool:
        """Run hardware simulation with output checking.

        Raises RuntimeError if outputs do not match the quantized recordings.
        Call save_files() before this method.

        Returns:
            True if all outputs match.
        """
        if self.files_dir is None:
            raise RuntimeError("Call save_files() before verify_hardware().")
        inputs_path = self.files_dir / "input_packets.npy"
        recordings_path = self.files_dir / "recordings.npy"
        self.simulation.run(inputs_path=inputs_path, recordings_path=recordings_path)
        print("[NIR2FPGA] Hardware output check PASSED.")
        return True

    def load_accelerator_output(
        self,
        ignore_timestamp: bool = False,
        vcd_path: Optional[str] = None,
    ) -> Any:
        """Decode the AXI output stream from the VCD. Delegates to simulation."""
        return self.simulation.load_accelerator_output(
            ignore_timestamp=ignore_timestamp,
            vcd_path=vcd_path,
        )
