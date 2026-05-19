"""
IOManager.py - PYNQ driver for sparse tensor to AXI4-Stream packet conversion

Converts sparse input tensors into 32-bit AXI packets for hardware accelerator.

Packet instruction format (bits [2:0]):
- 000: Timestep signal (end of timestep marker)
- 001: Spike arrival (bits [10:3] contain 8-bit coordinate)
"""

from __future__ import annotations

import numpy as np
import math
from typing import Any, Dict, Union, Tuple, List
from dataclasses import dataclass
import struct

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

@dataclass
class AcceleratorPacket:
    """
    32-bit packet for hardware accelerator.

    Instruction types:
    - 000 (0): Timestep signal (end of timestep)
    - 001 (1): Spike

    Packet structure:
    - Timestep
      - Bits [2:0]: 000
      - Bits [31:2]: unused
    - Spike
      - Bits [2:0]:   instruction type
      - Bits [15:3]:  coordinate (13 bits, for spike packets)
      - Bits [31:16]: spike value (quantized according to ConfigJSON)
    """

    instruction_type: int
    spike_value: int = 0
    spike_coordinate: int = 0

    # Instruction type constants
    TYPE_NOOP = 0b000      # No operation
    TYPE_SPIKE = 0b001     # Spike packet
    TYPE_TIMESTEP = 0b010  # Timestep marker

    def __post_init__(self):
        """Validate packet fields."""
        if self.instruction_type not in [self.TYPE_NOOP, self.TYPE_SPIKE, self.TYPE_TIMESTEP]:
            raise ValueError(
                f"Invalid instruction type {self.instruction_type}. "
                f"Must be {self.TYPE_NOOP} (noop), {self.TYPE_SPIKE} (spike), or {self.TYPE_TIMESTEP} (timestep)"
            )

        if self.spike_coordinate > 8191 or self.spike_coordinate < 0:
            raise ValueError(f"Coordinate {self.spike_coordinate} out of 13-bit range [0, 8191]")

        # Coordinate should only be set for spike packets
        if self.instruction_type == self.TYPE_TIMESTEP and self.spike_coordinate != 0:
            raise ValueError("Timestep packets must have spike_coordinate=0")


    def __int__(self) -> int:
        """
        Convert packet to 32-bit integer.

        Returns:
            32-bit integer with packed instruction and coordinate
        """
        packet = 0
        packet |= (self.instruction_type & 0b111)        # Bits [2:0]: type
        packet |= ((self.spike_coordinate & 0x1FFF) << 3)  # Bits [15:3]: coordinate (13 bits)
        packet |= ((self.spike_value & 0xFFFF) << 16)     # Bits [31:16]: spike_value
        return packet

    def __str__(self) -> str:
        """
        Pretty-print packet for debugging.

        Returns:
            Human-readable string representation
        """
        if self.instruction_type == self.TYPE_TIMESTEP:
            return f"[TIMESTEP] 0x{int(self):08x}"
        elif self.instruction_type == self.TYPE_SPIKE:
            return f"[SPIKE coord={self.spike_coordinate:03d}] 0x{int(self):08x}"
        else:
            return f"[UNKNOWN type={self.instruction_type}] 0x{int(self):08x}"

    def __repr__(self) -> str:
        """Representation for debugging."""
        return self.__str__()

    @classmethod
    def timestep_signal(cls) -> 'AcceleratorPacket':
        """Create a timestep signal packet."""
        return cls(instruction_type=cls.TYPE_TIMESTEP, spike_coordinate=0)

    @classmethod
    def noop(cls) -> 'AcceleratorPacket':
        """Create a NOOP (no operation) packet.

        NOOP packets are silently consumed by the hardware without
        affecting buffer state or timestep counting. Useful for:
        - Flow control: Keep AXI stream active
        - Debugging: Insert timing markers
        - Padding: Align packet streams

        Returns:
            AcceleratorPacket with instruction_type=TYPE_NOOP
        """
        return cls(instruction_type=cls.TYPE_NOOP, spike_coordinate=0)

    @classmethod
    def spike(cls, quantized_value: int, spike_coordinate: int) -> 'AcceleratorPacket':
        """Create a spike arrival packet."""
        return cls(instruction_type=cls.TYPE_SPIKE, spike_value=quantized_value, spike_coordinate=spike_coordinate)

    @classmethod
    def from_int(cls, value: int) -> 'AcceleratorPacket':
        instruction_type = value & 0b111
        spike_coordinate = (value >> 3) & 0x1FFF  # 13 bits
        spike_value = (value >> 16) & 0xFFFF      # Extract from bit 16
        return cls(instruction_type=instruction_type, spike_value=spike_value, spike_coordinate=spike_coordinate)


class IOManager:
    """
    Manages conversion of sparse tensors to AXI4-Stream packets for FPGA accelerator.

    Generates a flat array of AcceleratorPacket objects from sparse input tensors.
    Each timestep's spikes are followed by a timestep signal packet.
    """

    def __init__(self, input_shape: tuple[int, ...] | list[int], output_shape: tuple[int, ...] | list[int],
                 quant_data: Dict[str, Dict[str, Any]]) -> None:
        self.timesteps = input_shape[0]
        self.input_dims = list(input_shape[1:])
        self.input_shape = list(input_shape)
        self.input_quant_data = quant_data["input"]["output"]
        self.output_dims = list(output_shape)
        self.output_shape = [self.timesteps] + self.output_dims
        self.output_quant_data = quant_data["output"]["input"]

    def __call__(self, tensor: Any) -> List[AcceleratorPacket]:
        return self.create_input_packets(tensor)

    # Should be equivalent to QuantizationWrapper._quantize_tensor
    # - with no torch
    # - without converting back to floating point
    def quantize(self, x: Any, is_input: bool = True) -> int:
        if is_input:
            p = self.input_quant_data
        else:
            p = self.output_quant_data

        scale = 2 ** p["exp"]
        int_vals = int(math.floor(x / scale))
        int_vals = min(max(int_vals, p["min_value"]), p["max_value"])

        return int_vals

    def create_input_packets(self, tensor: Any) -> List[AcceleratorPacket]:
        # Safely extract shape from numpy array, torch tensor, or list
        if isinstance(tensor, np.ndarray):
            shape = tensor.shape
        elif isinstance(tensor, list):
            shape = (len(tensor),)
        elif torch is not None and isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
        else:
            raise ValueError(f"Unsupported tensor type: {type(tensor)}. Expected numpy.ndarray, list, or torch.Tensor.")

        if list(shape) != self.input_shape:
            raise ValueError(f"Expected shape {self.input_shape} but got {shape}.")

        all_packets = []

        for t in range(self.timesteps):
            timestep_data = np.asarray(tensor[t])
            nonzero_indices = np.argwhere(np.abs(timestep_data) > 0)

            for idx in nonzero_indices:
                if idx.size == 0:
                    continue

                quantized_val = self.quantize(float(timestep_data[tuple(idx)]))
                coord = self._flatten_coordinate(idx)
                packet = AcceleratorPacket.spike(
                    quantized_value=quantized_val,
                    spike_coordinate=coord
                )
                all_packets.append(packet)

            all_packets.append(AcceleratorPacket.timestep_signal())

        return all_packets

    def _flatten_coordinate(self, multi_idx: np.ndarray) -> int:
        """
        Convert multi-dimensional index to flattened 0D coordinate.

        Args:
            multi_idx: Multi-dimensional index array from np.argwhere()
                      Shape is always (ndim,) regardless of input dimensionality
                      E.g., [7] for 1D input, [5, 3] for 2D input

        Returns:
            Flattened coordinate as integer (0-255)

        Example:
            For shape (10,), index [7] -> 7
            For shape (28, 28), index [5, 3] -> 5*28 + 3 = 143
        """
        # np.argwhere always returns indices as rows in a 2D array
        # Each row has shape (ndim,) matching len(self.dims)

        # Safety checks
        if len(multi_idx) == 0 or multi_idx.size == 0:
            return 0

        # Flatten the index array to handle any shape
        flat_idx = multi_idx.flatten()

        # Ensure we don't exceed available dimensions
        num_dims = min(len(self.input_dims), len(flat_idx))

        # Handle 1D case: multi_idx is [coordinate]
        if len(self.input_dims) == 1:
            coord = int(flat_idx[0])
        else:
            # Multi-dimensional case: flatten in row-major order (C-style)
            coord = 0
            multiplier = 1
            for i in reversed(range(num_dims)):
                coord += int(flat_idx[i]) * multiplier
                multiplier *= self.input_dims[i]

        if coord > 8191:
            raise ValueError(
                f"Flattened coordinate {coord} exceeds 13-bit limit (8191). "
                f"Index: {multi_idx}, Dims: {self.input_dims}"
            )

        return int(coord)

    def create_input_packets_numpy(self, tensor: np.ndarray) -> np.ndarray:
        """Vectorized version of create_input_packets + packets_to_numpy.

        Returns a uint32 numpy array of packed AXI packets directly,
        bypassing AcceleratorPacket object creation entirely.

        Args:
            tensor: Input array of shape [timesteps, *input_dims]

        Returns:
            np.ndarray of uint32 packets ready for DMA transfer
        """
        tensor = np.asarray(tensor)
        if list(tensor.shape) != self.input_shape:
            raise ValueError(f"Expected shape {self.input_shape} but got {tensor.shape}.")

        p = self.input_quant_data
        scale = 2.0 ** p["exp"]
        min_val = p["min_value"]
        max_val = p["max_value"]
        is_1d = len(self.input_dims) == 1

        timestep_marker = np.array([np.uint32(AcceleratorPacket.TYPE_TIMESTEP)], dtype=np.uint32)
        chunks: List[np.ndarray] = []

        for t in range(self.timesteps):
            timestep_data = tensor[t]

            if is_1d:
                flat = timestep_data.ravel()
                coords = np.flatnonzero(np.abs(flat) > 0).astype(np.uint32)
                values = flat[coords]
            else:
                mask = np.abs(timestep_data) > 0
                indices = np.argwhere(mask)
                if len(indices) == 0:
                    chunks.append(timestep_marker)
                    continue
                coords = np.atleast_1d(np.ravel_multi_index(tuple(indices.T), self.input_dims)).astype(np.uint32)
                values = timestep_data[mask]

            if len(coords) > 0:
                int_vals = np.clip(
                    np.floor(values.astype(np.float64) / scale).astype(np.int32),
                    min_val, max_val
                )
                packets = (
                    np.uint32(AcceleratorPacket.TYPE_SPIKE)
                    | ((coords & np.uint32(0x1FFF)) << np.uint32(3))
                    | ((int_vals.astype(np.uint32) & np.uint32(0xFFFF)) << np.uint32(16))
                )
                chunks.append(packets)

            chunks.append(timestep_marker)

        return np.concatenate(chunks) if chunks else np.array([], dtype=np.uint32)

    def create_output_packets_numpy(self, tensor: np.ndarray) -> np.ndarray:
        """Vectorized encoder for output tensors in the AXI packet format.

        Expects a tensor of shape [timesteps, *output_dims] whose non-zero entries are
        already in the output quantization domain.
        """
        tensor = np.asarray(tensor)
        if list(tensor.shape) != self.output_shape:
            raise ValueError(f"Expected shape {self.output_shape} but got {tensor.shape}.")

        is_1d = len(self.output_dims) == 1
        timestep_marker = np.array([np.uint32(AcceleratorPacket.TYPE_TIMESTEP)], dtype=np.uint32)
        chunks: List[np.ndarray] = []

        for t in range(self.timesteps):
            timestep_data = tensor[t]

            if is_1d:
                flat = np.asarray(timestep_data).ravel()
                coords = np.flatnonzero(flat != 0).astype(np.uint32)
                values = flat[coords]
            else:
                mask = np.asarray(timestep_data) != 0
                indices = np.argwhere(mask)
                if len(indices) == 0:
                    chunks.append(timestep_marker)
                    continue
                coords = np.atleast_1d(np.ravel_multi_index(tuple(indices.T), self.output_dims)).astype(np.uint32)
                values = timestep_data[mask]

            if len(coords) > 0:
                if np.issubdtype(values.dtype, np.floating):
                    p = self.output_quant_data
                    scale = 2.0 ** p["exp"]
                    min_val = p["min_value"]
                    max_val = p["max_value"]
                    int_vals = np.clip(
                        np.floor(values.astype(np.float64) / scale).astype(np.int32),
                        min_val,
                        max_val,
                    )
                else:
                    int_vals = values.astype(np.int64)

                packets = (
                    np.uint32(AcceleratorPacket.TYPE_SPIKE)
                    | ((coords & np.uint32(0x1FFF)) << np.uint32(3))
                    | ((int_vals.astype(np.uint32) & np.uint32(0xFFFF)) << np.uint32(16))
                )
                chunks.append(packets)

            chunks.append(timestep_marker)

        return np.concatenate(chunks) if chunks else np.array([], dtype=np.uint32)

    def packets_to_numpy(self, packets: List[AcceleratorPacket]) -> np.ndarray:
        """
        Convert list of AcceleratorPacket to numpy array of 32-bit integers.

        Useful for sending to PYNQ DMA.

        Args:
            packets: List of AcceleratorPacket objects

        Returns:
            Numpy array of uint32 values ready for DMA transfer
        """
        return np.array([int(pkt) for pkt in packets], dtype=np.uint32)

    def decode_output_packets(self, packets: Union[List[int], np.ndarray]) -> np.ndarray:
        # Convert to numpy array if needed
        packets = np.array(packets, dtype=np.uint32)

        # 1. Create output array (reverse of starting with input tensor)
        output = np.zeros(self.output_shape, dtype=np.float32)

        # 2. Track current timestep (reverse of for t in range(self.timesteps))
        timestep_i = 0

        # 3. Loop over packets (reverse of creating packets)
        for packet in packets:
            # Decode packet fields (reverse of __int__ in AcceleratorPacket)
            instruction_type = int(packet & 0b111)
            coordinate = int((packet >> 3) & 0x1FFF)   # 13-bit coordinate
            spike_value = int((packet >> 16) & 0xFFFF) # Extract from bit 16

            if instruction_type == AcceleratorPacket.TYPE_TIMESTEP:  # 2
                # Reverse of: all_packets.append(AcceleratorPacket.timestep_signal())
                timestep_i += 1

                if timestep_i > self.timesteps:
                    print(f"warning: Received TIMESTEP #{timestep_i} but only {self.timesteps} expected")


            elif instruction_type == AcceleratorPacket.TYPE_SPIKE:  # 1
                # Reverse of: AcceleratorPacket.spike(quantized_value, spike_coordinate)

                # Bounds check
                if timestep_i >= self.timesteps:
                    raise ValueError(
                        f"Received SPIKE at timestep {timestep_i} >= {self.timesteps}"
                    )

                # 4. Dequantize value (reverse of self.quantize())
                dequantized_val = self.dequantize(spike_value)

                # 5. Unflatten coordinate (reverse of self._flatten_coordinate())
                multi_idx = self._unflatten_coordinate(coordinate)

                # 6. Set value in output tensor (reverse of reading from tensor[t])
                output[timestep_i][tuple(multi_idx)] = dequantized_val

            elif instruction_type == AcceleratorPacket.TYPE_NOOP:  # 0
                # Silently skip NOOPs
                pass

            else:
                raise ValueError(
                    f"[timestep {timestep_i}] Unknown instruction type {instruction_type} in packet 0x{packet:08x}"
                )

        return output

    def decode_output_packets_numpy(self, packets: Union[List[int], np.ndarray]) -> np.ndarray:
        """Vectorized version of decode_output_packets.

        Uses numpy operations instead of a Python for-loop.

        Args:
            packets: Raw uint32 packets from DMA

        Returns:
            Output tensor of shape [timesteps, *output_dims]
        """
        packets = np.asarray(packets, dtype=np.uint32)
        output = np.zeros(self.output_shape, dtype=np.float32)

        if len(packets) == 0:
            return output

        # Vectorized field extraction
        types = packets & np.uint32(0x7)
        coords = ((packets >> np.uint32(3)) & np.uint32(0x1FFF)).astype(np.int64)
        values = ((packets >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.int32)

        # Assign timestep index to each packet
        timestep_indices = np.flatnonzero(types == AcceleratorPacket.TYPE_TIMESTEP)
        spike_mask = types == AcceleratorPacket.TYPE_SPIKE
        spike_positions = np.flatnonzero(spike_mask)

        if len(spike_positions) == 0:
            return output

        # Spikes before first TIMESTEP are timestep 0, between 1st and 2nd are timestep 1, etc.
        spike_ts = np.searchsorted(timestep_indices, spike_positions, side='left')

        # Vectorized dequantization
        sv = values[spike_mask]
        p = self.output_quant_data
        if p["bits"] < 16:
            sv = sv & np.int32((1 << p["bits"]) - 1)
        if p.get("signed", False):
            sign_threshold = np.int32(2 ** (p["bits"] - 1))
            sv = np.where(sv >= sign_threshold, sv - np.int32(2 ** p["bits"]), sv)
        scale = 2.0 ** p["exp"]
        dequantized = sv.astype(np.float32) * np.float32(scale)

        # Assign to output
        spike_coords = coords[spike_mask]
        is_1d = len(self.output_dims) == 1
        if is_1d:
            output[spike_ts, spike_coords] = dequantized
        else:
            multi_idx = np.unravel_index(spike_coords, self.output_dims)
            output[(spike_ts,) + multi_idx] = dequantized

        return output

    def decode_output_packets_dual_numpy(self, packets: Union[List[int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Decode output packets into both dequantized and raw-signed tensors.

        Args:
            packets: Raw uint32 packets from DMA

        Returns:
            Tuple of:
              - dequantized_output: float32 tensor of shape [timesteps, *output_dims]
              - raw_output: int32 tensor of shape [timesteps, *output_dims]
        """
        packets = np.asarray(packets, dtype=np.uint32)
        dequantized_output = np.zeros(self.output_shape, dtype=np.float32)
        raw_output = np.zeros(self.output_shape, dtype=np.int32)

        if len(packets) == 0:
            return dequantized_output, raw_output

        # Vectorized field extraction
        types = packets & np.uint32(0x7)
        coords = ((packets >> np.uint32(3)) & np.uint32(0x1FFF)).astype(np.int64)
        values = ((packets >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.int32)

        # Assign timestep index to each packet
        timestep_indices = np.flatnonzero(types == AcceleratorPacket.TYPE_TIMESTEP)
        spike_mask = types == AcceleratorPacket.TYPE_SPIKE
        spike_positions = np.flatnonzero(spike_mask)

        if len(spike_positions) == 0:
            return dequantized_output, raw_output

        # Spikes before first TIMESTEP are timestep 0, between 1st and 2nd are timestep 1, etc.
        spike_ts = np.searchsorted(timestep_indices, spike_positions, side='left')

        # Convert wire-format unsigned payload to signed fixed-point integer domain.
        raw_signed = values[spike_mask]
        p = self.output_quant_data
        if p["bits"] < 16:
            raw_signed = raw_signed & np.int32((1 << p["bits"]) - 1)
        if p.get("signed", False):
            sign_threshold = np.int32(2 ** (p["bits"] - 1))
            raw_signed = np.where(raw_signed >= sign_threshold, raw_signed - np.int32(2 ** p["bits"]), raw_signed)

        # Dequantize from signed fixed-point integer domain.
        scale = 2.0 ** p["exp"]
        dequantized = raw_signed.astype(np.float32) * np.float32(scale)

        # Assign to outputs
        spike_coords = coords[spike_mask]
        is_1d = len(self.output_dims) == 1
        if is_1d:
            dequantized_output[spike_ts, spike_coords] = dequantized
            raw_output[spike_ts, spike_coords] = raw_signed
        else:
            multi_idx = np.unravel_index(spike_coords, self.output_dims)
            dequantized_output[(spike_ts,) + multi_idx] = dequantized
            raw_output[(spike_ts,) + multi_idx] = raw_signed

        return dequantized_output, raw_output

    def dequantize(self, spike_value: int) -> float:
        """
        Dequantize value (exact reverse of quantize()).

        quantize():   float_val → int_val = floor(float_val / scale)
        dequantize(): int_val → float_val = int_val * scale

        Args:
            spike_value: Quantized integer from hardware packet

        Returns:
            Dequantized float value
        """
        p = self.output_quant_data
        scale = 2 ** p["exp"]
        total_bits = p["bits"]

        spike_value = spike_value & ((2 ** total_bits) - 1)

        # Handle signed values - convert from unsigned to signed two's complement
        if p.get("signed", False):
            sign_threshold = 2 ** (total_bits - 1)
            if spike_value >= sign_threshold:
                spike_value = spike_value - (2 ** total_bits)

        # Exact reverse: multiply instead of divide
        float_val = spike_value * scale

        return float_val

    def _unflatten_coordinate(self, coord: int) -> np.ndarray:
        """
        Unflatten coordinate (exact reverse of _flatten_coordinate()).

        _flatten_coordinate():   multi_idx → coord (row-major order)
        _unflatten_coordinate(): coord → multi_idx

        Args:
            coord: Flattened coordinate (0-255)

        Returns:
            Multi-dimensional index matching self.output_dims

        Example:
            For output_dims = [1]: coord=0 → [0]
            For output_dims = [28, 28]: coord=143 → [5, 3]  (143 = 5*28 + 3)
        """
        if len(self.output_dims) == 1:
            # 1D case: coordinate is the index
            return np.array([coord])

        # Multi-dimensional case: reverse row-major flattening
        multi_idx = []
        for dim_size in reversed(self.output_dims):
            multi_idx.insert(0, coord % dim_size)
            coord //= dim_size

        return np.array(multi_idx)


if __name__ == "__main__":
    import argparse
    import sys
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="IOManager CLI for packet encoding/decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Encode input tensor to packets:
    python IOManager.py encode --model-dir ./model --input-file input.npy --output-file packets.npy

  Decode hardware output packets to tensor:
    python IOManager.py decode --model-dir ./model --input-file output_packets.npy --output-file output_tensor.npy
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # === ENCODE subcommand ===
    encode_parser = subparsers.add_parser("encode", help="Encode tensor to AXI packets")
    encode_parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Directory containing model.json and quantizations.json"
    )
    encode_parser.add_argument(
        "--input-file", type=str, required=True,
        help="Input tensor file (NPY format)"
    )
    encode_parser.add_argument(
        "--output-file", type=str, required=True,
        help="Output packets file (NPY uint32 format)"
    )

    # === DECODE subcommand ===
    decode_parser = subparsers.add_parser("decode", help="Decode AXI packets to tensor")
    decode_parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Directory containing model.json and quantizations.json"
    )
    decode_parser.add_argument(
        "--input-file", type=str, required=True,
        help="Input packets file (NPY uint32 format)"
    )
    decode_parser.add_argument(
        "--output-file", type=str, required=True,
        help="Output tensor file (NPY format)"
    )
    decode_parser.add_argument(
        "--raw", action="store_true",
        help="Output raw quantized values instead of dequantized floats"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        model_dir = Path(args.model_dir)

        # Load config
        model_json_path = model_dir / "model.json"
        quant_json_path = model_dir / "quantizations.json"

        if not model_json_path.exists():
            print(f"ERROR: model.json not found at {model_json_path}", file=sys.stderr)
            sys.exit(1)

        if not quant_json_path.exists():
            print(f"ERROR: quantizations.json not found at {quant_json_path}", file=sys.stderr)
            sys.exit(1)

        with open(model_json_path, "r") as f:
            model_data = json.load(f)

        with open(quant_json_path, "r") as f:
            quant_data = json.load(f)

        # Extract shapes
        input_shape = model_data["input"]["shape"]
        output_shape = model_data["output"]["shape"]

        # Create IOManager
        io_manager = IOManager(input_shape, output_shape, quant_data)

        if args.command == "encode":
            # Load input tensor
            input_file = Path(args.input_file)
            if not input_file.exists():
                print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
                sys.exit(1)

            input_tensor = np.load(input_file)
            print(f"✓ Loaded input tensor: shape={input_tensor.shape}, dtype={input_tensor.dtype}")

            # Generate packets
            packets = io_manager.create_input_packets_numpy(input_tensor)
            print(f"✓ Generated {len(packets)} packets")

            # Save packets
            output_file = Path(args.output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, packets)
            print(f"✓ Saved packets to {output_file}")
            print(f"  Packets file size: {packets.nbytes} bytes")

        elif args.command == "decode":
            # Load packets
            input_file = Path(args.input_file)
            if not input_file.exists():
                print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
                sys.exit(1)

            packets = np.load(input_file).astype(np.uint32)
            print(f"✓ Loaded {len(packets)} packets")

            # Decode packets
            if args.raw:
                dequantized, raw = io_manager.decode_output_packets_dual_numpy(packets)
                output = raw
                print(f"✓ Decoded packets (raw quantized values): shape={output.shape}")
            else:
                dequantized, raw = io_manager.decode_output_packets_dual_numpy(packets)
                output = dequantized
                print(f"✓ Decoded packets (dequantized): shape={output.shape}")

            # Save output
            output_file = Path(args.output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, output)
            print(f"✓ Saved output tensor to {output_file}")
            print(f"  Output file size: {output.nbytes} bytes")

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
