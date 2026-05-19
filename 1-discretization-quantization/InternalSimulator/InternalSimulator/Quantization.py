from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union
import types
from functools import partial
import math
import torch
import numpy as np

from .fp import FPTensor


def _optional_node_scalar(node: Any, field: str) -> Optional[float]:
    if not hasattr(node, field):
        return None
    value = getattr(node, field)
    return float(np.asarray(value).reshape(-1)[0])


def _to_param_tensor(value: Any) -> torch.Tensor:
    """Float32 tensor copy of a weight/bias, accepting numpy arrays or torch params.

    np.asarray() fails on a grad-enabled torch Parameter, so a torch tensor is
    detached directly instead of being round-tripped through numpy.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(torch.float32)
    return torch.tensor(np.asarray(value), dtype=torch.float32)


class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for fixed-point quantization.

    Forward: Apply real quantization (floor/clamp)
    Backward: Pass gradients through (identity), but zero out gradients for clipped values

    This enables gradient flow through quantization for Quantization Aware Training (QAT).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, min_val: int, max_val: int) -> torch.Tensor:
        """
        Quantize tensor to fixed-point representation.

        Args:
            x: Input tensor (float)
            scale: Quantization scale (2^exp, where exp is typically negative for frac_bits)
            min_val: Minimum quantized integer value
            max_val: Maximum quantized integer value

        Returns:
            Quantized tensor (float, but constrained to quantization levels)
        """
        # Real quantization in forward pass
        int_vals = torch.floor(x / scale)
        int_vals_clamped = torch.clamp(int_vals, min_val, max_val)
        output = int_vals_clamped * scale

        # Save bounds for backward pass (in float space)
        min_bound = min_val * scale
        max_bound = max_val * scale
        ctx.save_for_backward(x, torch.tensor([min_bound, max_bound], device=x.device, dtype=x.dtype))

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:  # type: ignore[override]
        """
        STE backward: pass gradients through, but zero out gradients for clipped values.

        This is "STE with clipping" - more stable than pure STE for training.
        """
        x, bounds = ctx.saved_tensors
        min_bound, max_bound = bounds[0], bounds[1]

        # Gradient is passed through where x is in valid range
        grad_input = grad_output.clone()
        grad_input[x < min_bound] = 0
        grad_input[x > max_bound] = 0

        # Return gradients for (x, scale, min_val, max_val)
        # Only x needs gradient; scale, min_val, max_val are constants
        return grad_input, None, None, None


class QuantizationWrapper(torch.nn.Module):
    def __init__(
        self,
        nir_node: Any,
        quants: Dict[str, Dict[str, Any]],
        use_ste: bool = False,
        spike_fn: Optional[Callable] = None,
        norm_input: bool = False,
        num_timesteps: int = 1,
    ) -> None:
        super().__init__()
        self.quants = quants
        self.use_ste = use_ste
        self.num_timesteps = num_timesteps
        self.record_states = True
        self.recordings: Dict[str, Any] = {}
        self.firing_rate: Optional[torch.Tensor] = None

        node_type = type(nir_node).__name__
        self.node_type = node_type
        self.layer_name = node_type  # kept for error messages in helpers

        match node_type:
            case "IF":
                self._init_neuron(nir_node, alpha_mem=1.0, spike_fn=spike_fn, norm_input=norm_input)
                self._neuron_forward = self.lif_layer_forward_quantized
            case "LIF" | "CubaLIF":
                tau_mem = float(np.asarray(nir_node.tau).reshape(-1)[0])
                self._init_neuron(nir_node, alpha_mem=1.0 - 1.0 / tau_mem, spike_fn=spike_fn, norm_input=norm_input)
                self._neuron_forward = self.lif_layer_forward_quantized
            case "LI":
                tau_mem = float(np.asarray(nir_node.tau).reshape(-1)[0])
                self._init_neuron(nir_node, alpha_mem=1.0 - 1.0 / tau_mem, spike_fn=None, norm_input=norm_input)
                self._neuron_forward = self.exp_leak_layer_forward_quantized
            case "I":
                self._init_neuron(nir_node, alpha_mem=1.0, spike_fn=None, norm_input=False)
                self._neuron_forward = self.exp_leak_layer_forward_quantized
            case "Linear" | "Affine":
                self.weight = torch.nn.Parameter(_to_param_tensor(nir_node.weight))
                bias_val = getattr(nir_node, "bias", None)
                if bias_val is not None:
                    self.bias: Optional[torch.nn.Parameter] = torch.nn.Parameter(
                        _to_param_tensor(bias_val)
                    )
                else:
                    self.bias = None
            case x:
                raise ValueError(
                    f"NIR2FPGA: NIR node type {x!r} has not been implemented for quantization yet"
                )

    def _init_neuron(
        self,
        nir_node: Any,
        alpha_mem: float,
        spike_fn: Optional[Callable],
        norm_input: bool,
    ) -> None:
        self.v_mem: Optional[torch.Tensor] = None
        self.i_syn: Optional[torch.Tensor] = None
        self.alpha_mem = alpha_mem
        tau_syn_val = _optional_node_scalar(nir_node, "tau_syn")
        self.alpha_syn: Optional[float] = (1.0 - 1.0 / tau_syn_val) if tau_syn_val is not None else None
        threshold_val = _optional_node_scalar(nir_node, "v_threshold")
        self.spike_threshold = torch.tensor(1.0 if threshold_val is None else threshold_val, dtype=torch.float32)
        self.min_v_mem: Optional[torch.Tensor] = None
        min_vm = _optional_node_scalar(nir_node, "min_v_mem")
        if min_vm is not None:
            self.min_v_mem = torch.tensor(min_vm, dtype=torch.float32)
        self.spike_fn = spike_fn
        self.surrogate_grad_fn: Optional[Callable] = None
        self.norm_input = norm_input

    def _ensure_state(self, batch_size: int, trailing_dim: list, device: torch.device) -> None:
        """Allocate neuron state on ``device``.

        State is (re)created when missing, when its shape no longer matches, or
        when it lives on a different device than the incoming activations — the
        last case keeps state consistent when the model runs on CUDA.
        """
        shape = (batch_size, *trailing_dim)
        if (
            self.v_mem is None
            or self.v_mem.shape != torch.Size(shape)
            or self.v_mem.device != device
        ):
            self.v_mem = torch.zeros(shape, device=device)
        if self.alpha_syn is not None and (
            self.i_syn is None
            or self.i_syn.shape != torch.Size(shape)
            or self.i_syn.device != device
        ):
            self.i_syn = torch.zeros(shape, device=device)

    def reset_states(self) -> None:
        """Clear neuron membrane/synaptic state.

        Called between independent inputs (e.g. each QAT batch) so state — and
        any autograd graph attached to it — does not leak across sequences.
        """
        self.v_mem = None
        self.i_syn = None

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        if self.node_type in {"IF", "LIF", "CubaLIF", "LI", "I"}:
            bt = input_data.shape[0]
            batch_size = bt // self.num_timesteps
            x = input_data.reshape(batch_size, self.num_timesteps, *input_data.shape[1:])
            out = self._neuron_forward(x)
            return out.reshape(bt, *out.shape[2:])
        else:
            out = self.linear_layer_forward_quantized(input_data)
            return out

    def quantize_fn(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], key: Optional[str] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(x, dict):
            return {k: self._quantize_tensor(v, k) for k, v in x.items()}

        if key is None:
            raise ValueError(
                "NIR2FPGA: Key must be provided when x is a tensor (not a dict)"
            )
        return self._quantize_tensor(x, key)

    def _quantize_tensor(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.quants.keys():
            raise ValueError(
                f"NIR2FPGA: Could not find {key} for layer {self.layer_name} in quantization parameters. Available keys: {list(self.quants.keys())}"
            )

        p = self.quants[key]
        scale = 2 ** p["exp"]
        min_val = p["min_value"]
        max_val = p["max_value"]

        if self.use_ste:
            # Use STE for backwardable quantization (QAT mode)
            result = STEQuantize.apply(x, scale, min_val, max_val)
            assert isinstance(result, torch.Tensor)
            return result
        else:
            # Hard quantization path uses saturating integer narrowing.
            int_vals = torch.floor(x / scale)
            raw = self._wrap_raw_tensor(int_vals.long(), p["bits"], p["signed"])
            return raw.float() * scale

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

    def _qu_tensor(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """Type-narrowed quantize_fn for when input is known to be a Tensor."""
        result = self.quantize_fn(x, key)
        assert isinstance(result, torch.Tensor)
        return result

    def _hard_quantize_tensor(self, x: torch.Tensor, key: str) -> torch.Tensor:
        p = self.quants[key]
        scale = 2 ** p["exp"]
        int_vals = torch.floor(x / scale)
        raw = self._wrap_raw_tensor(int_vals.long(), p["bits"], p["signed"])
        return raw.float() * scale

    def _quantize_to_int(self, x: torch.Tensor, key: str) -> torch.Tensor:
        p = self.quants[key]
        scale = 2 ** p["exp"]
        int_vals = torch.floor(x / scale)
        return self._wrap_raw_tensor(int_vals.long(), p["bits"], p["signed"])

    def _quantize_weights_to_int(self, x: torch.Tensor, key: str = "weights") -> torch.Tensor:
        """Quantize weights the way the Scala hardware materializes AFix constants.

        Empirically this matches truncation toward zero rather than floor for
        signed weight tensors.
        """
        p = self.quants[key]
        scale = 2 ** p["exp"]
        int_vals = torch.trunc(x / scale)
        return self._wrap_raw_tensor(int_vals.long(), p["bits"], p["signed"])

    def _dequantize_from_int(self, raw: torch.Tensor, key: str) -> torch.Tensor:
        scale = 2 ** self.quants[key]["exp"]
        return raw.float() * scale

    @staticmethod
    def _is_binary_spike_quant(quant: Dict[str, Any]) -> bool:
        return (
            quant["bits"] == 1
            and quant["min_value"] == 0
            and quant["max_value"] == 1
            and quant["exp"] == 0
            and not quant["signed"]
        )

    @staticmethod
    def _blend_hard_soft(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return soft + (hard - soft).detach()

    def _blend_state_dict(self, hard: dict, soft: dict) -> dict:
        return {key: self._blend_hard_soft(hard[key], soft[key]) for key in hard.keys()}

    def linear_forward_hard(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run Linear using hardware-like integer MAC semantics.

        This mirrors the Scala datapath more closely than linear_forward_soft:
        - input and weights are quantized to integer codes first
        - each product is rescaled to the output exponent before accumulation
        - accumulation happens on integer partial sums
        - final output is saturated and dequantized

        The implementation supports arbitrary leading dimensions, e.g.
        (..., in_features) -> (..., out_features).
        """
        input_raw = self._quantize_to_int(input_data, "input")
        weight_raw = self._quantize_weights_to_int(self.weight, "weights")

        input_shape = input_raw.shape
        in_features = input_shape[-1]
        out_features = weight_raw.shape[0]
        input_flat = input_raw.reshape(-1, in_features)

        exp_shift = self.quants["input"]["exp"] + self.quants["weights"]["exp"] - self.quants["output"]["exp"]

        out_q = self.quants["output"]
        accum_bits = out_q["bits"] + math.ceil(math.log2(max(1, in_features)))
        accum_raw = torch.zeros((input_flat.shape[0], out_features), dtype=torch.long, device=input_flat.device)
        chunk_size = min(64, in_features)

        if self._is_binary_spike_quant(self.quants["input"]):
            # Hardware materializes signed weights with truncation toward zero,
            # so keep the same behavior during exponent alignment.
            scaled_weight = self._wrap_raw_tensor(
                torch.trunc(weight_raw.float() * (2.0 ** exp_shift)).long(),
                out_q["bits"],
                out_q["signed"],
            )
            for start in range(0, in_features, chunk_size):
                stop = min(start + chunk_size, in_features)
                chunk = torch.nn.functional.linear(
                    input_flat[:, start:stop].float(),
                    scaled_weight[:, start:stop].float(),
                )
                chunk_raw = torch.floor(chunk).long()
                accum_raw = self._wrap_raw_tensor(accum_raw + chunk_raw, accum_bits, out_q["signed"])
        else:
            weight_float = weight_raw.float()
            for start in range(0, in_features, chunk_size):
                stop = min(start + chunk_size, in_features)
                scaled_products = torch.floor(
                    input_flat[:, start:stop].float().unsqueeze(1)
                    * weight_float[:, start:stop].unsqueeze(0)
                    * (2.0 ** exp_shift)
                ).long()
                chunk_raw = scaled_products.sum(dim=-1)
                accum_raw = self._wrap_raw_tensor(accum_raw + chunk_raw, accum_bits, out_q["signed"])

        if self.bias is not None:
            bias_raw = self._quantize_to_int(self.bias, "output")
            accum_raw = self._wrap_raw_tensor(accum_raw + bias_raw.unsqueeze(0), accum_bits, out_q["signed"])

        product_raw = self._wrap_raw_tensor(accum_raw, out_q["bits"], out_q["signed"])
        product_raw = product_raw.reshape(*input_shape[:-1], out_features)
        return self._dequantize_from_int(product_raw, "output")

    def linear_forward_soft(self, input_data: torch.Tensor) -> torch.Tensor:
        q_input = self._qu_tensor(input_data, "input")
        q_weight = self._qu_tensor(self.weight, "weights")
        bias = self.bias.data if self.bias is not None else None
        output = torch.nn.functional.linear(q_input, q_weight, bias)
        return self._qu_tensor(output, "output")

    def linear_layer_forward_quantized(self, input_data: torch.Tensor) -> torch.Tensor:
        hard_output = self.linear_forward_hard(input_data)
        if self.use_ste:
            soft_output = self.linear_forward_soft(input_data)
            output = self._blend_hard_soft(hard_output, soft_output)
        else:
            output = hard_output
        self.recordings = {}
        return output

    def to_fp(self, x: torch.Tensor, key: str) -> FPTensor:
        """
        Convert tensor to FPTensor using quantization config for key.

        This creates a true fixed-point representation that matches hardware
        arithmetic exactly, avoiding the float-then-floor pattern.
        """
        if key not in self.quants.keys():
            raise ValueError(
                f"NIR2FPGA: Could not find {key} for layer {self.layer_name} in quantization parameters. Available keys: {list(self.quants.keys())}"
            )
        p = self.quants[key]
        return FPTensor(x, p["int_bits"], p["frac_bits"], p["signed"])

    def _hard_coeff_mul(
        self,
        coeff: float,
        x: torch.Tensor,
        x_key: str,
        out_key: str,
        extra_frac_bits: int = 8,
    ) -> torch.Tensor:
        """Hardware-like multiply by a high-precision constant coefficient.

        Scala's Neuron path keeps leak/input-scaling coefficients in a format with
        extra fractional precision, multiplies by the quantized activation/state,
        then fixes the result back to the destination format. This helper mirrors
        that behavior for the software reference path.
        """
        x_raw = self._quantize_to_int(x, x_key).float()
        out_q = self.quants[out_key]
        x_q = self.quants[x_key]
        coeff_frac = out_q["frac_bits"] + extra_frac_bits
        coeff_raw = round(coeff * (2 ** coeff_frac))
        shift = coeff_frac + x_q["frac_bits"] - out_q["frac_bits"]
        prod_raw = torch.floor(x_raw * coeff_raw / (2.0 ** shift)).long()
        prod_raw = self._wrap_raw_tensor(prod_raw, out_q["bits"], out_q["signed"])
        return self._dequantize_from_int(prod_raw, out_key)

    def lif_forward_single_fp(
        self,
        input_fp: FPTensor,
        alpha_mem_fp: FPTensor,
        alpha_syn_fp: Optional[FPTensor],
        state_fp: dict,
        spike_threshold_fp: FPTensor,
        reset_value: float,
        spike_fn: Optional[Callable],
        min_v_mem_fp: Optional[FPTensor],
    ):
        """
        Single timestep LIF forward using true fixed-point arithmetic.

        This matches hardware behavior exactly by:
        1. Using FPTensor for all values (including alpha_mem!)
        2. Performing (a * b) >> frac_bits multiplication
        3. No float intermediates that get floored later
        """
        if alpha_syn_fp is not None:
            state_fp["i_syn"] = alpha_syn_fp * (state_fp["i_syn"] + input_fp)
            syn_input = state_fp["i_syn"]
        else:
            syn_input = input_fp

        # Integrate: v_mem = alpha_mem * v_mem + input
        # In hardware this is: leaked = alpha_mem * v_mem, then integrated = leaked + input
        leaked = alpha_mem_fp * state_fp["v_mem"]
        integrated = leaked + syn_input

        # Threshold check
        if spike_fn is not None:
            will_spike = integrated >= spike_threshold_fp

            # Reset: where spike, set to reset_value; else keep integrated
            reset_fp = self.to_fp(
                torch.full(integrated.shape, reset_value, dtype=torch.float32),
                "v_mem"
            )
            state_fp["v_mem"] = FPTensor.where(will_spike, reset_fp, integrated)

            # Output spikes as float tensor (0.0 or 1.0)
            spikes = will_spike.float()
        else:
            # No spiking (LI neuron) - just output the membrane potential
            state_fp["v_mem"] = integrated
            spikes = integrated.to_float()

        # Apply min_v_mem if specified
        if min_v_mem_fp is not None:
            # v_mem = max(v_mem, min_v_mem)
            below_min = state_fp["v_mem"] < min_v_mem_fp
            state_fp["v_mem"] = FPTensor.where(below_min, min_v_mem_fp, state_fp["v_mem"])

        return spikes, state_fp

    def lif_forward_fp(
        self,
        input_data: torch.Tensor,
        alpha_mem: float,
        alpha_syn: Optional[float],
        tau_mem: float,
        dt: float,
        state: dict,
        spike_threshold: torch.Tensor,
        spike_fn: Optional[Callable],
        reset_value: float,
        min_v_mem: Optional[torch.Tensor],
        norm_input: bool = False,
        record_states: bool = False,
    ):
        """
        Full LIF forward pass using hardware-like fixed-point arithmetic.

        For the common alpha_syn=None case used by the Scala hardware path, this
        mirrors Neuron.scala directly:
        - coefficients are kept at higher precision than v_mem
        - input/leak products are separately fixed to the v_mem format
        - the sum is then quantized back to v_mem
        """
        if alpha_syn is None:
            state_q = {
                key: self._hard_quantize_tensor(val, key if key in self.quants else "v_mem")
                for key, val in state.items()
            }
            threshold_q = self._hard_quantize_tensor(spike_threshold, "v_mem")
            min_v_mem_q = self._hard_quantize_tensor(min_v_mem, "v_mem") if min_v_mem is not None else None
            input_factor = (1.0 - alpha_mem) if norm_input else 1.0

            n_time_steps = input_data.shape[1]
            output_spikes = []
            recordings = {name: [] for name in state_q.keys()} if record_states else {}

            for step in range(n_time_steps):
                leaked = self._hard_coeff_mul(alpha_mem, state_q["v_mem"], "v_mem", "v_mem")
                input_contrib = self._hard_coeff_mul(input_factor, input_data[:, step], "input", "v_mem")
                updated = self._hard_quantize_tensor(leaked + input_contrib, "v_mem")

                if spike_fn is not None:
                    will_spike = updated >= threshold_q
                    reset_tensor = self._hard_quantize_tensor(torch.full_like(updated, reset_value), "v_mem")
                    state_q["v_mem"] = torch.where(will_spike, reset_tensor, updated)
                    spikes = will_spike.float()
                else:
                    state_q["v_mem"] = updated
                    spikes = updated.clone()

                if min_v_mem_q is not None:
                    state_q["v_mem"] = torch.maximum(state_q["v_mem"], min_v_mem_q)

                output_spikes.append(spikes)
                if record_states:
                    for name in state_q.keys():
                        recordings[name].append(state_q[name].clone())

            record_dict = {
                name: torch.stack(vals, 1) for name, vals in recordings.items()
            } if record_states else {}
            return torch.stack(output_spikes, 1), state_q, record_dict

        to_fp = self.to_fp
        alpha_mem_fp = to_fp(torch.tensor([alpha_mem], dtype=torch.float32), "v_mem")
        alpha_syn_fp = to_fp(torch.tensor([alpha_syn], dtype=torch.float32), "i_syn")

        input_scale: Optional[float] = None
        if norm_input:
            input_scale = 1.0 - alpha_mem

        state_fp = {}
        for key, val in state.items():
            if key in self.quants:
                state_fp[key] = to_fp(val, key)
            else:
                state_fp[key] = to_fp(val, "v_mem")

        spike_threshold_fp = to_fp(spike_threshold, "v_mem")
        min_v_mem_fp = to_fp(min_v_mem, "v_mem") if min_v_mem is not None else None

        n_time_steps = input_data.shape[1]
        state_names = list(state_fp.keys())
        output_spikes = []
        recordings = {name: [] for name in state_names} if record_states else {}

        for step in range(n_time_steps):
            if norm_input and input_scale is not None:
                input_fp = to_fp(input_data[:, step] * input_scale, "v_mem")
            else:
                input_fp = to_fp(input_data[:, step], "v_mem")

            spikes, state_fp = self.lif_forward_single_fp(
                input_fp=input_fp,
                alpha_mem_fp=alpha_mem_fp,
                alpha_syn_fp=alpha_syn_fp,
                state_fp=state_fp,
                spike_threshold_fp=spike_threshold_fp,
                reset_value=reset_value,
                spike_fn=spike_fn,
                min_v_mem_fp=min_v_mem_fp,
            )
            output_spikes.append(spikes)

            if record_states:
                for name in state_names:
                    recordings[name].append(state_fp[name].to_float().clone())

        state_float = {k: v.to_float() for k, v in state_fp.items()}
        record_dict = {
            name: torch.stack(vals, 1) for name, vals in recordings.items()
        } if record_states else {}
        return torch.stack(output_spikes, 1), state_float, record_dict

    def lif_forward_single_quantized(
        self,
        input_data: torch.Tensor,
        alpha_mem: Any,
        alpha_syn: Any,
        state: dict[str, Any],
        spike_threshold: Any,
        spike_fn: Any,
        reset_fn: Any,
        surrogate_grad_fn: Any,
        min_v_mem: Any,
        norm_input: bool,
    ) -> Any:

        qu = self._qu_tensor

        # if t_syn was provided, we're going to use synaptic current dynamics
        if alpha_syn is not None:
            state["i_syn"] = qu(alpha_syn * (state["i_syn"] + input_data), "v_mem")
        else:
            state["i_syn"] = qu(input_data, "input")
        if norm_input:
            synaptic_input = qu((1 - alpha_mem) * state["i_syn"], "v_mem")
        else:
            synaptic_input = state["i_syn"]
            # Decay the membrane potential and add the input currents which are normalised by tau

        state["v_mem"] = qu(alpha_mem * state["v_mem"] + synaptic_input, "v_mem")

        # generate spikes and adjust v_mem
        if spike_fn:
            input_tensors = [state[name] for name in spike_fn.required_states]
            spikes = spike_fn.apply(*input_tensors, spike_threshold, surrogate_grad_fn)
            state = self.quantize_fn(reset_fn(spikes, state, spike_threshold), "v_mem")  # type: ignore[assignment]
            # Quantize v_mem after reset
            state["v_mem"] = qu(state["v_mem"], "v_mem")
        else:
            v_mem_tensor: torch.Tensor = state["v_mem"]  # type: ignore[assignment]
            spikes = v_mem_tensor.clone()
            state = state.copy()
        if min_v_mem is not None:
            state["v_mem"] = qu(
                torch.nn.functional.relu(state["v_mem"] - min_v_mem) + min_v_mem,
                "v_mem",
            )
        return spikes, state

    def lif_forward_quantized(
        self,
        input_data: torch.Tensor,
        alpha_mem: Any,
        alpha_syn: Any,
        state: dict[str, Any],
        spike_threshold: Any,
        spike_fn: Any,
        reset_fn: Any,
        surrogate_grad_fn: Any,
        min_v_mem: Any,
        norm_input: bool,
        record_states: bool = False,
    ) -> Any:
        qu = self._qu_tensor

        # Quantize initial state
        state = {key: qu(val, key) for key, val in state.items()}

        # Quantize input data
        input_data = qu(input_data, "input")

        # Quantize constants
        alpha_mem = alpha_mem.item()
        alpha_syn = alpha_syn.detach().clone() if alpha_syn is not None else None
        spike_threshold = qu(spike_threshold.detach().clone(), "v_mem")
        min_v_mem = (
            qu(min_v_mem.detach().clone(), "v_mem") if min_v_mem is not None else None
        )

        n_time_steps = input_data.shape[1]
        state_names = list(state.keys())
        output_spikes = []
        recordings: dict[str, list[Any]] = {}
        if record_states:
            recordings = {name: [] for name in state_names}

        for step in range(n_time_steps):
            spikes, state = self.lif_forward_single_quantized(
                input_data=input_data[:, step],
                alpha_mem=alpha_mem,
                alpha_syn=alpha_syn,
                state=state,
                spike_threshold=spike_threshold,
                spike_fn=spike_fn,
                reset_fn=reset_fn,
                surrogate_grad_fn=surrogate_grad_fn,
                min_v_mem=min_v_mem,
                norm_input=norm_input,
            )
            output_spikes.append(spikes)
            if record_states:
                for name in state_names:
                    recordings[name].append(state[name].clone())

        if record_states:
            record_dict = {
                name: torch.stack(vals, 1) for name, vals in recordings.items()
            }
        else:
            record_dict = dict()

        return torch.stack(output_spikes, 1), state, record_dict

    def lif_forward_single_qat(
        self,
        input_data: torch.Tensor,
        alpha_mem: float,
        state: dict,
        spike_threshold: torch.Tensor,
        spike_fn: Optional[Callable],
        surrogate_grad_fn: Optional[Callable],
        min_v_mem: Optional[torch.Tensor],
    ):
        """
        Single timestep LIF forward for QAT training.

        Uses float tensors with STE quantization at key points.
        This allows gradients to flow through the quantization operations.

        Key differences from lif_forward_single_quantized:
        - Keeps all tensors as float (no FPTensor)
        - Uses STE-enabled quantize_fn for backwardable quantization
        - Spike function uses surrogate gradients (already backwardable)
        """
        qu = self._qu_tensor

        # Leak: v_mem = alpha_mem * v_mem
        leaked = qu(alpha_mem * state["v_mem"], "v_mem")

        # Integrate: v_mem = leaked + input
        integrated = qu(leaked + input_data, "v_mem")

        # Spike generation with surrogate gradient
        if spike_fn is not None:
            input_tensors = [integrated]  # v_mem is the input to spike function
            spikes = spike_fn.apply(*input_tensors, spike_threshold, surrogate_grad_fn)  # type: ignore[union-attr]

            # Reset: v_mem = 0 where spike, else keep integrated
            # Use straight-through for the reset selection
            reset_value = torch.zeros_like(integrated)
            state["v_mem"] = torch.where(spikes > 0.5, reset_value, integrated)
            state["v_mem"] = qu(state["v_mem"], "v_mem")
        else:
            # No spiking (LI neuron) - output membrane potential
            spikes = integrated
            state["v_mem"] = integrated

        # Apply min_v_mem if specified
        if min_v_mem is not None:
            state["v_mem"] = qu(
                torch.nn.functional.relu(state["v_mem"] - min_v_mem) + min_v_mem,
                "v_mem",
            )

        return spikes, state

    def lif_forward_qat(
        self,
        input_data: torch.Tensor,
        alpha_mem: float,
        state: dict,
        spike_threshold: torch.Tensor,
        spike_fn: Optional[Callable],
        surrogate_grad_fn: Optional[Callable],
        min_v_mem: Optional[torch.Tensor],
        record_states: bool = False,
    ):
        """
        Full LIF forward pass for QAT training.

        Uses float tensors with STE quantization - all gradients flow through.

        Args:
            input_data: Input tensor (batch, time, neurons)
            alpha_mem: Membrane decay constant
            state: Dictionary containing 'v_mem' tensor
            spike_threshold: Threshold for spike generation
            spike_fn: Spike function (with surrogate gradient)
            surrogate_grad_fn: Surrogate gradient function for backprop
            min_v_mem: Optional minimum membrane voltage
            record_states: Whether to record internal states

        Returns:
            (output_spikes, final_state, recordings)
        """
        qu = self._qu_tensor

        # Quantize initial state with STE
        state = {key: qu(val, key) for key, val in state.items()}

        # Quantize input data with STE
        input_data = qu(input_data, "input")

        # Quantize threshold (with STE for consistency)
        spike_threshold = qu(spike_threshold, "v_mem")

        # Quantize min_v_mem if provided
        if min_v_mem is not None:
            min_v_mem = qu(min_v_mem, "v_mem")

        n_time_steps = input_data.shape[1]
        state_names = list(state.keys())
        output_spikes = []

        if record_states:
            recordings = {name: [] for name in state_names}
        else:
            recordings = {}

        for step in range(n_time_steps):
            spikes, state = self.lif_forward_single_qat(
                input_data=input_data[:, step],
                alpha_mem=alpha_mem,
                state=state,
                spike_threshold=spike_threshold,
                spike_fn=spike_fn,
                surrogate_grad_fn=surrogate_grad_fn,
                min_v_mem=min_v_mem,
            )
            output_spikes.append(spikes)

            if record_states:
                for name in state_names:
                    recordings[name].append(state[name].clone())

        if record_states:
            record_dict = {
                name: torch.stack(vals, 1) for name, vals in recordings.items()
            }
        else:
            record_dict = {}

        return torch.stack(output_spikes, 1), state, record_dict

    def exp_leak_layer_forward_quantized(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_data: Data to be processed. Expected shape: (batch, time, ...)

        Returns:
            Membrane potential with same shape as `input_data`.
        """
        batch_size, time_steps, *trailing_dim = input_data.shape
        device = input_data.device
        self._ensure_state(batch_size, trailing_dim, device)
        assert self.v_mem is not None

        min_v_mem = self.min_v_mem.to(device) if self.min_v_mem is not None else None
        tau_mem = 1.0 / (1.0 - self.alpha_mem) if self.alpha_mem != 1.0 else float("inf")
        hard_output, hard_state, hard_recordings = self.lif_forward_fp(
            input_data=input_data,
            alpha_mem=self.alpha_mem,
            alpha_syn=None,
            tau_mem=tau_mem,
            dt=1.0,
            state={"v_mem": self.v_mem.clone()},
            spike_threshold=torch.zeros(1, device=device),
            spike_fn=None,
            reset_value=0.0,
            min_v_mem=min_v_mem,
            norm_input=self.norm_input,
            record_states=self.record_states,
        )

        if self.use_ste:
            soft_output, soft_state, _ = self.lif_forward_qat(
                input_data=input_data,
                alpha_mem=self.alpha_mem,
                state={"v_mem": self.v_mem.clone()},
                spike_threshold=torch.zeros(1, device=device),
                spike_fn=None,
                surrogate_grad_fn=None,
                min_v_mem=min_v_mem,
                record_states=self.record_states,
            )
            output = self._blend_hard_soft(hard_output, soft_output)
            state = self._blend_state_dict(hard_state, soft_state)
            recordings = hard_recordings
        else:
            output, state, recordings = hard_output, hard_state, hard_recordings

        self.v_mem = state["v_mem"]
        self.recordings = recordings
        self.firing_rate = output.mean()
        return output

    def current_filter_layer_forward_quantized(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, *trailing_dim = input_data.shape
        self._ensure_state(batch_size, trailing_dim, input_data.device)
        assert self.v_mem is not None

        alpha_mem = self.alpha_mem
        qu = self._qu_tensor

        state_hard = qu(self.v_mem, "v_mem")
        input_hard = qu(input_data, "input")
        hard_outputs = []
        hard_recordings: Dict[str, list] = {"v_mem": []} if self.record_states else {}

        for step in range(input_hard.shape[1]):
            leaked = self._hard_coeff_mul(alpha_mem, state_hard, "v_mem", "v_mem")
            input_contrib = self._hard_coeff_mul(alpha_mem, input_hard[:, step], "input", "v_mem")
            state_hard = self._hard_quantize_tensor(leaked + input_contrib, "v_mem")
            hard_outputs.append(state_hard.clone())
            if self.record_states:
                hard_recordings["v_mem"].append(state_hard.clone())

        hard_output = torch.stack(hard_outputs, 1)

        if self.use_ste:
            state_soft = qu(self.v_mem, "v_mem")
            input_soft = qu(input_data, "input")
            soft_outputs = []
            for step in range(input_soft.shape[1]):
                leaked = qu(alpha_mem * state_soft, "v_mem")
                input_contrib = qu(alpha_mem * input_soft[:, step], "v_mem")
                state_soft = qu(leaked + input_contrib, "v_mem")
                soft_outputs.append(state_soft.clone())
            soft_output = torch.stack(soft_outputs, 1)
            output = self._blend_hard_soft(hard_output, soft_output)
            final_state = self._blend_hard_soft(state_hard, state_soft)
        else:
            output = hard_output
            final_state = state_hard

        self.v_mem = final_state
        self.recordings = (
            {"v_mem": torch.stack(hard_recordings["v_mem"], 1)} if self.record_states else {}
        )
        self.firing_rate = output.mean()
        return output

    def lif_layer_forward_quantized(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_data: Data to be processed. Expected shape: (batch, time, ...)

        Returns:
            Output data with same shape as `input_data`.
        """
        batch_size, time_steps, *trailing_dim = input_data.shape
        device = input_data.device
        self._ensure_state(batch_size, trailing_dim, device)
        assert self.v_mem is not None

        alpha_syn = self.alpha_syn
        tau_mem = 1.0 / (1.0 - self.alpha_mem) if self.alpha_mem != 1.0 else float("inf")
        min_v_mem = self.min_v_mem.to(device) if self.min_v_mem is not None else None

        state: Dict[str, torch.Tensor] = {"v_mem": self.v_mem.clone()}
        if alpha_syn is not None and self.i_syn is not None:
            state["i_syn"] = self.i_syn.clone()

        hard_spikes, hard_state, hard_recordings = self.lif_forward_fp(
            input_data=input_data,
            alpha_mem=self.alpha_mem,
            alpha_syn=alpha_syn,
            tau_mem=tau_mem,
            dt=1.0,
            state=state,
            spike_threshold=self.spike_threshold.detach().clone().to(device),
            spike_fn=self.spike_fn,
            reset_value=0.0,
            min_v_mem=min_v_mem,
            norm_input=self.norm_input,
            record_states=self.record_states,
        )

        if self.use_ste:
            soft_spikes, soft_state, _ = self.lif_forward_qat(
                input_data=input_data,
                alpha_mem=self.alpha_mem,
                state={"v_mem": self.v_mem.clone()},
                spike_threshold=self.spike_threshold.to(device),
                spike_fn=self.spike_fn,
                surrogate_grad_fn=self.surrogate_grad_fn,
                min_v_mem=min_v_mem,
                record_states=self.record_states,
            )
            spikes = self._blend_hard_soft(hard_spikes, soft_spikes)
            final_state = self._blend_state_dict(hard_state, soft_state)
            recordings = hard_recordings
        else:
            spikes, final_state, recordings = hard_spikes, hard_state, hard_recordings

        self.v_mem = final_state["v_mem"]
        self.i_syn = final_state.get("i_syn")
        self.recordings = recordings
        self.firing_rate = spikes.mean()
        return spikes


class QuantizationConfig:
    def __init__(
        self,
        recordings: Dict[str, torch.Tensor],
        nir_node: Any,
        total_bits: int,
        weight_bits: Optional[int] = None,
        use_ste: bool = False,
        calibration_method: str = "minmax",
        calibration_percentile: Optional[float] = None,
        threshold_headroom_multiplier: float = 2.0,
        readout_percentile: Optional[float] = None,
        percentile_bounds: Optional[Dict[str, tuple[float, float]]] = None,
        readout_percentile_bounds: Optional[Dict[str, tuple[float, float]]] = None,
        spike_fn: Optional[Callable] = None,
        norm_input: bool = False,
        num_timesteps: int = 1,
    ) -> None:
        self.nir_node = nir_node
        self.quants = self._define_quant_dict(
            recordings,
            total_bits=total_bits,
            weight_bits=weight_bits,
            calibration_method=calibration_method,
            calibration_percentile=calibration_percentile,
            threshold_headroom_multiplier=threshold_headroom_multiplier,
            readout_percentile=readout_percentile,
            percentile_bounds=percentile_bounds,
            readout_percentile_bounds=readout_percentile_bounds,
        )
        self.use_ste = use_ste
        self.wrapper = QuantizationWrapper(
            nir_node,
            self.quants,
            use_ste=use_ste,
            spike_fn=spike_fn,
            norm_input=norm_input,
            num_timesteps=num_timesteps,
        )

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if key not in self.quants.keys():
            raise KeyError(
                f"No quantization config found for key ''{key}''. Available keys: {self.quants.keys()}"
            )

        return self.quants[key]

    def keys(self) -> Any:
        return self.quants.keys()

    def _define_quant_dict(
        self,
        recordings: Dict[str, torch.Tensor],
        total_bits: int,
        weight_bits: Optional[int] = None,
        calibration_method: str = "minmax",
        calibration_percentile: Optional[float] = None,
        threshold_headroom_multiplier: float = 2.0,
        readout_percentile: Optional[float] = None,
        percentile_bounds: Optional[Dict[str, tuple[float, float]]] = None,
        readout_percentile_bounds: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        PTQ procedure assuming dynamic fixed-point (AFix in SpinalHDL).

        From `recordings`, derive the values required to instantiate an AFix quantization in SpinalHDL:
        https://spinalhdl.github.io/SpinalDoc-RTD/master/SpinalHDL/Data%20types/AFix.html

        Procedure:
        1. From `recordings`, extract all unique values for each parameter.
        2. Extract min, max, sign.
        3. Calculate int_bits with `ceil(log₂(max_magnitude + 1))`
        4. Calculate frac_bits with `target_bits - int_bits`
        5. Calculate exponent with `exp = -frac_bits`
        6. Clamp to `target_bits`

        Args:
            recordings: Dictionary of recordings {parameter (str): value (torch.Tensor)}x
            total_bits: Fixed bit-width for quantization

        Returns:
            Dict with {'min_value', 'max_value', 'exp', 'bits', 'ideal_bits'}
        """
        # 1. Unique values
        unique_values = {}
        for param_name, recording in zip(recordings.keys(), recordings.values()):
            unique_values[param_name] = recording.unique()

        quant_dict = {}
        for param_name, recording in unique_values.items():
            target_bits = weight_bits if param_name == "weights" and weight_bits is not None else total_bits

            # Handle the single-bit (spike) in/out case
            if param_name in ["input", "output"] and torch.all(
                (recording == 0) | (recording == 1)
            ):
                quant_dict[param_name] = {
                    "min_value": int(0),
                    "max_value": int(1),
                    "exp": 0,
                    "bits": 1,
                    "int_bits": 1,
                    "frac_bits": 0,
                    "signed": False,
                }
                continue

            # 2. Min, Max, Sign
            data_min = recording.min().item()
            data_max = recording.max().item()

            if percentile_bounds is not None and param_name in percentile_bounds:
                bounded_min, bounded_max = percentile_bounds[param_name]
                data_min = float(bounded_min)
                data_max = float(bounded_max)

            if calibration_method == "percentile":
                if not (percentile_bounds is not None and param_name in percentile_bounds):
                    percentile = 99.9 if calibration_percentile is None else calibration_percentile
                    if not 0.0 < percentile <= 100.0:
                        raise ValueError(
                            f"calibration_percentile must be in (0, 100], got {percentile}"
                        )
                    flat = recording.detach().flatten().float()
                    if flat.numel() > 0:
                        tail = (100.0 - percentile) / 2.0
                        lower_q = tail / 100.0
                        upper_q = 1.0 - lower_q
                        data_min = torch.quantile(flat, lower_q).item()
                        data_max = torch.quantile(flat, upper_q).item()
            elif calibration_method != "minmax":
                raise NotImplementedError(
                    f"Calibration method '{calibration_method}' is not implemented."
                )

            if (
                readout_percentile is not None
                and self.nir_node is not None
                and type(self.nir_node).__name__ in {"LI"}
                and param_name in {"v_mem", "output"}
            ):
                if readout_percentile_bounds is not None and param_name in readout_percentile_bounds:
                    bounded_min, bounded_max = readout_percentile_bounds[param_name]
                    data_min = float(bounded_min)
                    data_max = float(bounded_max)
                else:
                    if not 0.0 < readout_percentile <= 100.0:
                        raise ValueError(
                            f"readout_percentile must be in (0, 100], got {readout_percentile}"
                        )
                    flat = recording.detach().flatten().float()
                    if flat.numel() > 0:
                        tail = (100.0 - readout_percentile) / 2.0
                        lower_q = tail / 100.0
                        upper_q = 1.0 - lower_q
                        data_min = torch.quantile(flat, lower_q).item()
                        data_max = torch.quantile(flat, upper_q).item()

            if data_max < data_min:
                data_max = data_min

            # Special handling for v_mem in LIF layers: ensure it can represent up to spike threshold
            if param_name == "v_mem" and self.nir_node is not None:
                nir_type = type(self.nir_node).__name__
                if nir_type in {"LIF", "CubaLIF", "IF"}:
                    threshold_val = _optional_node_scalar(self.nir_node, "v_threshold")
                    spike_thresh: float = 1.0 if threshold_val is None else threshold_val
                    data_max = max(data_max, spike_thresh * threshold_headroom_multiplier)

            is_signed = data_min < 0

            # 3. Integer bits
            if is_signed:
                max_magnitude = max(abs(data_min), abs(data_max))
                int_bits = math.ceil(math.log2(max_magnitude + 1)) + 1  # +1 for sign
            else:
                int_bits = math.ceil(math.log2(data_max + 1)) if data_max > 0 else 1

            # 4. Fractional bits
            frac_bits = target_bits - int_bits
            if frac_bits < 0:
                module_name = type(self.nir_node).__name__ if self.nir_node is not None else "unknown"
                raise ValueError(
                    f"Observed quantization range cannot be represented by provided bit budget of {target_bits} "
                    f"(module '{module_name}', parameter '{param_name}', min={data_min}, max={data_max}, "
                    f"needs {int_bits} integer bits, leaving frac_bits={frac_bits})"
                )
            exp = -frac_bits

            # 5. Exponent
            if is_signed:
                min_value = -(1 << (target_bits - 1))
                max_value = (1 << (target_bits - 1)) - 1
            else:
                min_value = 0
                max_value = (1 << target_bits) - 1

            quant_dict[param_name] = {
                "min_value": int(min_value),
                "max_value": int(max_value),
                "exp": exp,
                "bits": target_bits,
                "int_bits": int_bits,
                "frac_bits": frac_bits,
                "signed": is_signed,
            }
        return quant_dict

    def to_qformat(self, param: str) -> str:
        qc = self.quants[param]
        prefix = "SQ" if qc["signed"] else "UQ"
        return f"{prefix}{qc['int_bits']}.{qc['frac_bits']}"
