from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import nir
import numpy as np
import setVCD

Factory = Callable[[int], np.ndarray]

# NIR node type → hardware component name (set via setName in types/*.scala)
_NIR_TYPE_TO_HW_NAME: Dict[type, str] = {
    nir.LIF: "lif",
    nir.LI:  "li",
    nir.IF:  "if",
    nir.I:   "i",
}

# ── Section A: NIR type → recordable parameter mapping ────────────────────────

# Spiking neuron types: expose membrane voltage probe + spike output
_SPIKING_NEURON_TYPES = (nir.LIF, nir.IF, nir.CubaLIF)
# Leaky integrator types: continuous output only (no spike threshold)
_LEAKY_INTEGRATOR_TYPES = (nir.LI, nir.I)
# Linear transform types
_LINEAR_TYPES = (nir.Affine, nir.Linear)


def get_node_parameters(node: nir.NIRNode) -> List[str]:
    """Return the recordable hardware signal names for a given NIR node."""
    if isinstance(node, _SPIKING_NEURON_TYPES):
        return ["v_mem", "output"]
    if isinstance(node, _LEAKY_INTEGRATOR_TYPES):
        return ["output"]
    if isinstance(node, _LINEAR_TYPES):
        return ["output"]
    return []


def is_hardware_node(node: nir.NIRNode) -> bool:
    """Return True if this NIR node has observable hardware signals."""
    return bool(get_node_parameters(node))


# ── Section B: HW map discovery ───────────────────────────────────────────────

def _sorted_layer_ids(nir_graph: nir.NIRGraph, node_types: tuple) -> List[str]:  # type: ignore[type-arg]
    """Return numeric node keys (sorted) whose NIR type matches node_types."""
    return sorted(
        (k for k, v in nir_graph.nodes.items() if isinstance(v, node_types) and k.isdigit()),
        key=int,
    )


def _hw_paths_for(vcd: Any, prefix: str) -> List[str]:
    """
    Return sorted list of VCD component paths for a given name prefix.

    SpinalHDL setName() convention: the first instance uses the bare name (e.g. "li"),
    and subsequent instances are numbered (e.g. "li_1", "li_2").
    """
    paths: List[str] = []
    if vcd.search(f"{prefix}\\."):
        paths.append(prefix)
    paths += [f"{prefix}_{n}" for n in range(1, 100) if vcd.search(f"{prefix}_{n}")]
    return paths


def discover_hw_maps(
    vcd: Any,
    nir_graph: nir.NIRGraph,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return (layer_to_neuron_map, layer_to_linear_map) by searching the VCD
    for type-named components (lif, li_1, affine, affine_1, …).

    Both maps store layer_id → full VCD component path (e.g. "li", "li_1").
    """
    linear_layer_ids = _sorted_layer_ids(nir_graph, _LINEAR_TYPES)

    layer_to_neuron: Dict[str, str] = {}
    for nir_type, hw_name in _NIR_TYPE_TO_HW_NAME.items():
        ids   = _sorted_layer_ids(nir_graph, (nir_type,))
        paths = _hw_paths_for(vcd, hw_name)
        if len(paths) < len(ids):
            raise ValueError(
                f"Found {len(paths)} '{hw_name}' components in VCD but model has "
                f"{len(ids)} {nir_type.__name__} layers. Re-run the hardware simulation."
            )
        layer_to_neuron.update({lid: path for lid, path in zip(ids, paths)})

    affine_paths = _hw_paths_for(vcd, "affine")
    if len(affine_paths) < len(linear_layer_ids):
        raise ValueError(
            f"Found {len(affine_paths)} 'affine' components in VCD but model has "
            f"{len(linear_layer_ids)} linear layers. Re-run the hardware simulation."
        )
    layer_to_linear: Dict[str, str] = {
        lid: path for lid, path in zip(linear_layer_ids, affine_paths)
    }

    return layer_to_neuron, layer_to_linear


# ── Section C: Signal resolution ─────────────────────────────────────────────

def _search_required(vcd: Any, pattern: str, label: str) -> str:
    results = vcd.search(pattern)
    if not results:
        raise KeyError(
            f"[VCDMapping] Required signal not found: '{label}' (pattern: '{pattern}'). "
            "Please ensure VCD signal names correspond to lowercase NIR primitives, parameters, and 'input' / 'output' interfaces."
        )
    return results[0]


def _search_optional(vcd: Any, pattern: str) -> Optional[str]:
    results = vcd.search(pattern)
    return results[0] if results else None


def resolve_global_signals(vcd: Any) -> Dict[str, str]:
    """
    Resolve clock, reset, timestamp, and AXI output signals.
    Used by load_accelerator_output(). Replaces the global-signal block in
    NIR2FPGA.make_vcd_signals().
    """
    return {
        "clock":        _search_required(vcd, "AcceleratorAXI.clk",          "clock"),
        "reset":        _search_required(vcd, "AcceleratorAXI.reset",         "reset"),
        "timestamp":    _search_required(vcd, r"configTimestamp\[31:0\]",     "timestamp"),
        "m_axis_valid": _search_required(vcd, "m_axis_tvalid",                "m_axis_valid"),
        "m_axis_ready": _search_required(vcd, "m_axis_tready",                "m_axis_ready"),
        "m_axis_data":  _search_required(vcd, r"AcceleratorAXI\.m_axis_tdata","m_axis_data"),
    }


def resolve_neuron_signals(vcd: Any, component_path: str) -> Dict[str, Optional[str]]:
    """
    Resolve all observable signals for the given VCD component path (e.g. "li", "lif_1").
    Replaces the neuron branch of NIR2FPGA.make_vcd_signals().
    """
    p = component_path
    return {
        "clock":    _search_required(vcd, "AcceleratorAXI.clk",  "clock"),
        "reset":    _search_required(vcd, "AcceleratorAXI.reset", "reset"),
        "o_valid":  _search_required(vcd, f"{p}.output_valid",          f"{p}.output_valid"),
        "o_ready":  _search_required(vcd, f"{p}.output_ready",          f"{p}.output_ready"),
        "o_last":   _search_required(vcd, f"{p}.output_payload_last",   f"{p}.output_payload_last"),
        "o_payload":_search_required(vcd, f"{p}.output_payload_fragment_value_0",
                                         f"{p}.output_payload_fragment_value_0"),
        "o_coords": _search_optional(vcd, f"{p}.output_payload_fragment_coords"),
        "i_valid":  _search_optional(vcd, f"{p}.input_valid"),
        "i_ready":  _search_optional(vcd, f"{p}.input_ready"),
        "i_last":   _search_optional(vcd, f"{p}.input_payload_last"),
        "i_coords": _search_optional(vcd, f"{p}.input_payload_fragment_coords_0"),
    }


def resolve_linear_signals(
    vcd: Any, component_path: str
) -> Dict[str, Optional[str]]:
    """
    Resolve all observable signals for the given VCD component path (e.g. "affine", "affine_1").
    Note: o_last and payload use underscore convention (not dot).
    """
    p = component_path
    return {
        "clock":    _search_required(vcd, "AcceleratorAXI.clk",  "clock"),
        "reset":    _search_required(vcd, "AcceleratorAXI.reset", "reset"),
        "o_valid":  _search_required(vcd, f"{p}.output_valid",         f"{p}.output_valid"),
        "o_ready":  _search_required(vcd, f"{p}.output_ready",         f"{p}.output_ready"),
        "o_last":   _search_required(vcd, f"{p}_output_payload_last",  f"{p}_output_payload_last"),
        "o_payload":_search_required(vcd, f"{p}_output_payload_fragment_value_0",
                                         f"{p}_output_payload_fragment_value_0"),
        "o_coords": _search_optional(vcd, f"{p}_output_payload_fragment_coords_0"),
    }


# ── Section D: Factory functions ──────────────────────────────────────────────

def _make_fp_type(quant: Dict[str, Any]) -> Any:
    """Build setVCD.FP from a quantization dict {bits, frac_bits, signed}."""
    fp_ctor = cast(Any, setVCD.FP)
    fp_sig = inspect.signature(setVCD.FP)
    if "total_bits" in fp_sig.parameters:
        return fp_ctor(total_bits=quant["bits"], frac=quant["frac_bits"], signed=quant["signed"])
    return fp_ctor(frac=quant["frac_bits"], signed=quant["signed"])


def _build_handshake(vcd: Any, sigs: Dict[str, Optional[str]]) -> Any:
    """Return rising_edge(clock) & reset=0 & o_valid=1 & o_ready=1 expression."""
    rising = vcd.get(sigs["clock"], lambda x, y: x == 0 and y == 1)
    rst0   = vcd.get(sigs["reset"], lambda x: x == 0)
    valid  = vcd.get(sigs["o_valid"], lambda x: x == 1)
    ready  = vcd.get(sigs["o_ready"], lambda x: x == 1)
    return rising & rst0 & valid & ready


def neuron_v_mem_factory(
    vcd: Any,
    component_path: str,
    quant_v_mem: Dict[str, Any],
    sigs: Dict[str, Optional[str]],
    probe_count: int,
    timesteps: int,
) -> Factory:
    """
    Return Factory: index -> (timesteps,) numpy array of v_mem values.

    Resolves v_mem probe signal names at construction time.
    Coordinate-aggregates at call time: for each timestep, the last value
    where coord == index is taken.

    Raises IndexError for index >= probe_count (hardware limit is 10 probes).

    Signal naming convention: the hardware `v_mem` debug Vec holds one
    `Neuron.State` per memory word, and each `State.value` is a Vec of `width`
    lanes (neurons are packed `width` per word). Neuron `index` is therefore
    probe word `index // width`, lane `index % width` — VCD signal
    `{comp}.v_mem_{word}_value_{lane}`. `width` is discovered from the VCD.
    """
    handshake = _build_handshake(vcd, sigs)
    fp_type   = _make_fp_type(quant_v_mem)
    comp      = component_path

    # Discover the lane count (State.value Vec width) from probe word 0.
    width = 0
    while vcd.search(fr"{comp}\.v_mem_0(_value).*_{width}\b"):
        width += 1
    if width == 0:
        raise KeyError(
            f"[VCDMapping] no v_mem probe signals found under '{comp}' "
            f"(expected '{comp}.v_mem_0(_value).*_0')"
        )

    # Resolve the probe signal for each neuron index once at construction.
    probe_signals: List[str] = []
    for index in range(probe_count):
        word, lane = divmod(index, width)
        pattern = fr"{comp}\.v_mem_{word}_value_{lane}\b"
        sig = vcd.search(pattern)
        if not sig:
            raise KeyError(
                f"[VCDMapping] v_mem probe signal not found: {pattern}"
            )
        probe_signals.append(sig[0])

    o_last_sig   = sigs["o_last"]
    o_coords_sig = sigs["o_coords"]

    def factory(index: int) -> np.ndarray:
        if index >= probe_count:
            raise IndexError(
                f"{comp} has only {probe_count} v_mem debug probe(s) "
                f"(hardware limit is 10); requested index {index}."
            )
        all_v_mem  = vcd.get_values(probe_signals[index], handshake, value_type=fp_type)
        all_o_last = vcd.get_values(o_last_sig, handshake)
        all_coords = (
            vcd.get_values(o_coords_sig, handshake) if o_coords_sig
            else [index] * len(all_v_mem)
        )
        result = np.zeros(timesteps, dtype=float)
        t = 0
        for v, coord, last in zip(all_v_mem, all_coords, all_o_last):
            # Offset by one timestep (debug is async)
            if coord == index and t > 0:
                result[t - 1] = v
            if last == 1:
                t += 1
        return result

    return factory


def neuron_output_factory(
    vcd: Any,
    quant_out: Dict[str, Any],
    sigs: Dict[str, Optional[str]],
    num_neurons: int,
    timesteps: int,
) -> Factory:
    """
    Return Factory: index -> (timesteps,) numpy array of output values.

    Reads o_payload at handshake times; sums contributions where coord == index
    per timestep (o_last marks timestep boundaries).

    Raises IndexError for index >= num_neurons.

    Moves: NIR2FPGA._get_neuron_outputs() for neuron-type layers.
    """
    handshake    = _build_handshake(vcd, sigs)
    fp_type      = _make_fp_type(quant_out)
    o_last_sig   = sigs["o_last"]
    o_payload_sig= sigs["o_payload"]
    o_coords_sig = sigs["o_coords"]

    def factory(index: int) -> np.ndarray:
        if index >= num_neurons:
            raise IndexError(
                f"Layer has {num_neurons} neurons; requested index {index}."
            )
        all_payloads = vcd.get_values(o_payload_sig, handshake, value_type=fp_type)
        all_o_last   = vcd.get_values(o_last_sig, handshake)
        all_coords   = (
            vcd.get_values(o_coords_sig, handshake) if o_coords_sig
            else [0] * len(all_payloads)
        )
        result = np.zeros(timesteps, dtype=float)
        t = 0
        for payload, coord, last in zip(all_payloads, all_coords, all_o_last):
            if coord == index:
                result[t] += payload
            if last == 1:
                t += 1
        return result

    return factory


def linear_output_factory(
    vcd: Any,
    quant_out: Dict[str, Any],
    sigs: Dict[str, Optional[str]],
    num_outputs: int,
    timesteps: int,
) -> Factory:
    """
    Return Factory: index -> (timesteps,) numpy array of output values.

    Same aggregation logic as neuron_output_factory; signal names come from
    resolve_linear_signals() rather than resolve_neuron_signals().

    Raises IndexError for index >= num_outputs.

    Moves: NIR2FPGA._get_neuron_outputs() for Linear layers.
    """
    handshake    = _build_handshake(vcd, sigs)
    fp_type      = _make_fp_type(quant_out)
    o_last_sig   = sigs["o_last"]
    o_payload_sig= sigs["o_payload"]
    o_coords_sig = sigs["o_coords"]

    def factory(index: int) -> np.ndarray:
        if index >= num_outputs:
            raise IndexError(
                f"Linear layer has {num_outputs} outputs; requested index {index}."
            )
        all_payloads = vcd.get_values(o_payload_sig, handshake, value_type=fp_type)
        all_o_last   = vcd.get_values(o_last_sig, handshake)
        all_coords   = (
            vcd.get_values(o_coords_sig, handshake) if o_coords_sig
            else [0] * len(all_payloads)
        )
        result = np.zeros(timesteps, dtype=float)
        t = 0
        for payload, coord, last in zip(all_payloads, all_coords, all_o_last):
            if coord == index:
                result[t] += payload
            if last == 1:
                t += 1
        return result

    return factory
