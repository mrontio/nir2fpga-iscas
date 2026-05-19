from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import nir
import numpy as np
import torch  # needed for internal_model type hint

import InternalSimulator.VCDMapping as VCDMapping

Factory              = Callable[[int], np.ndarray]
HardwareExpressionMap = Dict[Tuple[str, str], Factory]
HardwareRecordings   = Dict[Tuple[str, int, str], np.ndarray]


class HardwareVariables:
    """
    Lazy, cached access to per-index VCD signal recordings.

    Constructed by HardwareSimulation.simulate() after the VCD file is loaded.
    Factories are built eagerly in __init__ — they capture setvcd in closures
    but do not read signal values yet.  Actual VCD reads happen on the first
    call to get_recording() for a given (node, index, parameter) triple;
    results are cached in HardwareRecordings.
    """

    def __init__(
        self,
        setvcd: Any,
        nir_graph: nir.NIRGraph,
        internal_model: torch.nn.Sequential,
        quantization_data: Dict[str, Any],
        timesteps: int,
    ) -> None:
        self._expression_map: HardwareExpressionMap = {}
        self._cache: HardwareRecordings = {}
        self._node_ids: List[str] = []
        self._param_map: Dict[str, List[str]] = {}

        layer_to_neuron, layer_to_linear = VCDMapping.discover_hw_maps(setvcd, nir_graph)
        self.layer_to_neuron_map = layer_to_neuron
        self.layer_to_linear_map = layer_to_linear

        for layer_id, node in nir_graph.nodes.items():
            params = VCDMapping.get_node_parameters(node)
            if not params:
                continue

            self._node_ids.append(layer_id)
            self._param_map[layer_id] = params

            sinabs_layer = internal_model[int(layer_id)]
            quant        = quantization_data[layer_id]

            if isinstance(node, VCDMapping._LINEAR_TYPES):
                sigs        = VCDMapping.resolve_linear_signals(setvcd, layer_to_linear[layer_id])
                num_outputs = int(sinabs_layer.out_features)  # type: ignore[union-attr]
                self._expression_map[(layer_id, "output")] = VCDMapping.linear_output_factory(
                    vcd=setvcd,
                    quant_out=quant["output"],
                    sigs=sigs,
                    num_outputs=num_outputs,
                    timesteps=timesteps,
                )

            else:
                component_path = layer_to_neuron[layer_id]
                sigs           = VCDMapping.resolve_neuron_signals(setvcd, component_path)
                num_neurons    = int(sinabs_layer.shape[1])  # type: ignore[union-attr]

                self._expression_map[(layer_id, "output")] = VCDMapping.neuron_output_factory(
                    vcd=setvcd,
                    quant_out=quant["output"],
                    sigs=sigs,
                    num_neurons=num_neurons,
                    timesteps=timesteps,
                )

                if isinstance(node, VCDMapping._SPIKING_NEURON_TYPES):
                    probe_count = min(num_neurons, 10)
                    if num_neurons > 10:
                        print(
                            f"[HardwareVariables] Layer {layer_id} has {num_neurons} neurons "
                            f"but only 10 v_mem debug probes are available."
                        )
                    self._expression_map[(layer_id, "v_mem")] = VCDMapping.neuron_v_mem_factory(
                        vcd=setvcd,
                        component_path=component_path,
                        quant_v_mem=quant["v_mem"],
                        sigs=sigs,
                        probe_count=probe_count,
                        timesteps=timesteps,
                    )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_nodes(self) -> List[str]:
        """Layer IDs for all hardware-observable layers."""
        return list(self._node_ids)

    def get_parameters(self, node: str) -> List[str]:
        """Recordable parameter names for a given layer ID."""
        if node not in self._param_map:
            raise KeyError(f"Node '{node}' is not a hardware-observable layer.")
        return list(self._param_map[node])

    def get_recording(
        self,
        node: str,
        index: int,
        parameters: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return a dict mapping parameter name → (timesteps,) numpy array.

        If parameters is None, all recordable parameters for the node are returned.
        First call per (node, index, parameter) evaluates the VCD factory and caches
        the result; subsequent calls return the cached array.

        Raises KeyError for unknown node or parameter.
        Raises IndexError for out-of-range index (e.g. v_mem index >= 10).
        """
        resolved = parameters if parameters is not None else self.get_parameters(node)
        result: Dict[str, np.ndarray] = {}
        for parameter in resolved:
            cache_key = (node, index, parameter)
            if cache_key in self._cache:
                result[parameter] = self._cache[cache_key]
                continue

            factory_key = (node, parameter)
            if factory_key not in self._expression_map:
                available = [f"('{n}', '{p}')" for n, p in self._expression_map]
                raise KeyError(
                    f"No hardware recording available for node='{node}', "
                    f"parameter='{parameter}'. Available: {available}"
                )

            value = self._expression_map[factory_key](index)
            self._cache[cache_key] = value
            result[parameter] = value
        return result

    def clear_cache(self) -> None:
        """Evict all cached recordings, freeing memory."""
        self._cache.clear()
