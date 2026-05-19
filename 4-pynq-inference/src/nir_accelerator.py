try:
    from pynq import Device, Overlay, allocate, MMIO
    import pynq.lib.dma as DMA
except ImportError as e:
    raise EnvironmentError(
        "❌ Not running in a PYNQ environment. Required libraries not found."
    ) from e

import os
import queue
import numpy as np
import json
import time
import math
import threading
import importlib
from pathlib import Path
from typing import Tuple, Union
import IOManager


class NIRAccelerator:
    """
    Simplified PYNQ accelerator for NIR-based SNN models.

    Usage:
        accel = NIRAccelerator("model.json", "design_1")
        output = accel(input_array)
    """

    def __init__(
            self,
            path: str,
            dma_timeout_sec: float = 5.0,
            debug: bool = False
    ):
        """
        Initialize NIRAccelerator.

        Args:
            path: Path to directory with quantizations.json, input.npy, output.npy, overlay.bit and overlay.hwh
            dma_timeout_sec: DMA operation timeout in seconds (default: 5.0)
        """
        # Load quantization config
        self.base_path = Path(path)
        with open(self.base_path / "quantizations.json", 'r') as f:
            quant_json = json.load(f)

        self.quant_data = quant_json["quantizations"]
        self.timesteps = quant_json["timesteps"]

        # Derive shapes from sample npy files (shape: [batch, timesteps, *dims])
        input_data = np.load(self.base_path / "input.npy")
        input_shape = list(input_data.shape[1:])  # Remove batch dimension only

        output_data = np.load(self.base_path / "output.npy")
        output_shape = list(output_data.shape[2:])  # Remove batch and timesteps dimensions

        # Load overlay

        bit_files = list(self.base_path.glob("*.bit"))
        if len(bit_files) == 0:
            raise FileNotFoundError(f"No .bit file found in {self.base_path}")
        elif len(bit_files) > 1:
            raise ValueError(f"Multiple .bit files found in {self.base_path}: {bit_files}")

        self.debug = debug

        try:
            devices = Device.devices
        except RuntimeError as e:
            raise RuntimeError(
                "Could not enumerate a usable PYNQ device in this shell. "
                "This usually means the shell environment does not have access to the FPGA "
                "device node or XRT/driver stack, even though the same overlay may work in Jupyter."
            ) from e

        if len(devices) == 0:
            raise RuntimeError(
                "No PYNQ devices were discovered. The overlay cannot be loaded until a device "
                "is visible to the current process."
            )

        if self.debug:
            print(f"[NIRAccelerator] Detected PYNQ devices: {devices}")

        self.overlay = Overlay(str(bit_files[0]), device=devices[0])
        if self.debug:
            print(f"[NIRAccelerator] Overlay loaded: {bit_files[0]}")
        self.dma = self.overlay.axi_dma_0  # Assuming DMA is named axi_dma_0
        self.dma_timeout_sec = dma_timeout_sec

        # AXI-Lite management register map on AcceleratorAXI
        self.REG_INPUT_TS = 0x00
        self.REG_OUTPUT_TS = 0x04

        # Resolve AcceleratorAXI control MMIO from overlay metadata
        self.accel_name = "AcceleratorAXI_0"
        if self.accel_name not in self.overlay.ip_dict:
            matching_ips = [
                name for name in self.overlay.ip_dict
                if "acceleratoraxi" in name.lower()
            ]
            if len(matching_ips) == 1:
                self.accel_name = matching_ips[0]
            elif len(matching_ips) > 1:
                raise RuntimeError(
                    f"Multiple AcceleratorAXI-like IPs found: {matching_ips}. "
                    "Set self.accel_name explicitly."
                )
            else:
                available_ips = list(self.overlay.ip_dict.keys())
                raise RuntimeError(
                    "AcceleratorAXI control IP not found in overlay.ip_dict. "
                    f"Expected '{self.accel_name}' or name containing 'acceleratoraxi'. "
                    f"Available IPs: {available_ips}"
                )

        accel_info = self.overlay.ip_dict[self.accel_name]
        self.accel_mmio = MMIO(accel_info["phys_addr"], accel_info["addr_range"])

        self.start_time = None

        # Create IOManager for packet encoding/decoding
        self.io_manager = IOManager.IOManager(
            input_shape=input_shape,
            output_shape=output_shape,
            quant_data=self.quant_data
        )

        # Extract buffer sizes
        self.num_input_neurons = int(np.prod(self.io_manager.input_dims))
        self.num_output_neurons = int(np.prod(self.io_manager.output_dims))
        self._max_expected_spike_packets = self.timesteps * self.num_output_neurons
        self._max_expected_output_words = self.timesteps * (self.num_output_neurons + 1)

        # self.buffer_size = 4095  # Maximum words per transfer
        self.buffer_size = 4095

        # Allocate buffers once in constructor for reuse
        self.input_buffer = allocate(shape=(self.buffer_size,), dtype=np.uint32)
        self.output_buffer = allocate(shape=(self.buffer_size,), dtype=np.uint32)

        # Pre-allocate output collection array for expected packet budget with one DMA chunk slack.
        max_output_words = self._max_expected_output_words + self.buffer_size
        self._output_array = np.empty(max_output_words, dtype=np.uint32)

        # Persistent worker threads — started once, reused across all inferences
        self._send_queue: queue.Queue = queue.Queue()
        self._recv_queue: queue.Queue = queue.Queue()
        self._send_done: queue.Queue = queue.Queue()
        self._recv_done: queue.Queue = queue.Queue()
        self._shutdown = threading.Event()

        self._send_thread = threading.Thread(target=self._send_loop, daemon=True, name="DMA-Send")
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True, name="DMA-Recv")
        self._send_thread.start()
        self._recv_thread.start()

    def _send_loop(self):
        """Persistent send worker — blocks on queue between inferences."""
        while not self._shutdown.is_set():
            try:
                work = self._send_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            input_packets, num_transfers, receiving = work
            error = None
            try:
                offset = 0
                for i in range(num_transfers):
                    end = min(offset + self.buffer_size, len(input_packets))
                    chunk = input_packets[offset:end]
                    self.input_buffer[0:len(chunk)] = chunk
                    nbytes = len(chunk) * 4
                    if not receiving.wait(timeout=self.dma_timeout_sec):
                        raise RuntimeError(
                            f"Send: Receive not ready after {self.dma_timeout_sec}s "
                            f"(chunk {i+1}/{num_transfers})"
                        )
                    self.dma.sendchannel.transfer(self.input_buffer, nbytes=nbytes)
                    self.dma.sendchannel.wait()
                    if self.debug:
                        elapsed = time.time() - self.start_time if self.start_time else 0.0
                        print(f"[NIRAccelerator] Sent chunk {i+1}/{num_transfers} ({len(chunk)} words) at +{elapsed:.3f}s")
                    offset = end
            except Exception as e:
                error = e
            self._send_done.put(error)

    def _recv_loop(self):
        """Persistent recv worker — blocks on queue between inferences."""
        while not self._shutdown.is_set():
            try:
                work = self._recv_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            expected_timesteps, receiving = work
            error = None
            received_timesteps = 0
            write_pos = 0
            total_spike_packets = 0
            try:
                while received_timesteps < expected_timesteps:
                    receiving.set()
                    self.dma.recvchannel.transfer(self.output_buffer)
                    self.dma.recvchannel.wait()
                    receiving.clear()

                    status = self.s2mm_status()
                    if status["dma_int_err"] or status["dma_slv_err"] or status["dma_dec_err"]:
                        raise RuntimeError(f"DMA error: {status}")

                    num_recv = status["length_packets"]
                    if num_recv == 0:
                        continue

                    # Grow collection buffer if stream exceeds expected budget.
                    needed = write_pos + num_recv
                    if needed > self._output_array.shape[0]:
                        new_capacity = max(self._output_array.shape[0] * 2, needed + self.buffer_size)
                        grown = np.empty(new_capacity, dtype=np.uint32)
                        grown[:write_pos] = self._output_array[:write_pos]
                        self._output_array = grown

                    chunk = self.output_buffer[0:num_recv]
                    self._output_array[write_pos:write_pos + num_recv] = chunk

                    if self.debug:
                        print(f"Received {num_recv} raw packets: ")
                        for data in chunk:
                            print(f"\t{data}")
                        print()

                    types = chunk & np.uint32(0x7)
                    batch_ts = int(np.sum(
                        types == IOManager.AcceleratorPacket.TYPE_TIMESTEP
                    ))
                    batch_spikes = int(np.sum(types == IOManager.AcceleratorPacket.TYPE_SPIKE))

                    if batch_spikes > 0:
                        spike_mask = (types == IOManager.AcceleratorPacket.TYPE_SPIKE)
                        spike_words = chunk[spike_mask]
                        spike_coords = ((spike_words >> np.uint32(3)) & np.uint32(0x1FFF))
                        if spike_coords.size > 0:
                            max_idx = int(np.argmax(spike_coords))
                            max_coord = int(spike_coords[max_idx])
                            if max_coord >= self.num_output_neurons:
                                offending_word = int(spike_words[max_idx])
                                raise RuntimeError(
                                    "Packet contract violation: output spike coordinate out of range "
                                    f"(max coord {max_coord}, expected < {self.num_output_neurons}, "
                                    f"offending word 0x{offending_word:08x}, "
                                    f"timesteps_rcvd={received_timesteps}, batch_words={num_recv}, "
                                    f"overlay_path='{self.base_path}')."
                                )

                    total_spike_packets += batch_spikes
                    max_spikes_for_run = expected_timesteps * self.num_output_neurons
                    if total_spike_packets > max_spikes_for_run:
                        raise RuntimeError(
                            "Packet contract violation: received too many spike packets "
                            f"({total_spike_packets} > {max_spikes_for_run}) for "
                            f"{expected_timesteps} timesteps and {self.num_output_neurons} output neurons."
                        )

                    write_pos += num_recv
                    received_timesteps += batch_ts
                    if self.debug and batch_ts > 0:
                        elapsed = time.time() - self.start_time if self.start_time else 0.0
                        print(f"[NIRAccelerator] Received timestep {received_timesteps}/{expected_timesteps} at +{elapsed:.3f}s")
            except Exception as e:
                error = e
                receiving.set()  # Unblock send thread
            self._recv_done.put((error, write_pos))

    def sendreceive(self, input_packets: np.ndarray, expected_timesteps: int) -> np.ndarray:
        """
        Send all input packets and receive all output packets via persistent threads.

        Args:
            input_packets: All input packets as numpy array
            expected_timesteps: Number of TIMESTEP packets to expect in output

        Returns:
            All output packets as numpy array
        """
        receiving = threading.Event()
        num_transfers = math.ceil(len(input_packets) / self.buffer_size)

        # Enqueue work for persistent threads
        self._recv_queue.put((expected_timesteps, receiving))
        self._send_queue.put((input_packets, num_transfers, receiving))

        # Wait for both to complete
        send_error = self._send_done.get()
        recv_error, write_pos = self._recv_done.get()

        if send_error:
            raise RuntimeError(f"Send failed: {send_error}")
        if recv_error:
            raise RuntimeError(f"Recv failed: {recv_error}")

        return self._output_array[:write_pos].copy()

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on the accelerator.

        Args:
            input_array: Input numpy array of shape [timesteps, *input_dims]

        Returns:
            Output numpy array of shape [timesteps, *output_dims]
        """
        # Vectorized packet encoding (replaces create_input_packets + packets_to_numpy)
        input_packets = self.io_manager.create_input_packets_numpy(input_array)
        return self.infer_from_packets(input_packets)

    def infer_from_packets(self, input_packets: np.ndarray) -> np.ndarray:
        """
        Run inference from pre-encoded packets (skips encoding step).

        Useful when packets are pre-computed by a PrefetchingLoader.

        Args:
            input_packets: uint32 numpy array of packed AXI packets

        Returns:
            Output numpy array of shape [timesteps, *output_dims]
        """
        expected_timesteps = int(np.sum(
            (input_packets & 0x7) == IOManager.AcceleratorPacket.TYPE_TIMESTEP
        ))

        self.start_time = time.time()
        output_packets = self.sendreceive(input_packets, expected_timesteps)

        # Vectorized packet decoding
        return self.io_manager.decode_output_packets_numpy(output_packets)

    def infer_from_packets_with_raw(self, input_packets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference from pre-encoded packets and return both output domains.

        Args:
            input_packets: uint32 numpy array of packed AXI packets

        Returns:
            Tuple of:
              - dequantized_output: float32 tensor [timesteps, *output_dims]
              - raw_output: int32 tensor [timesteps, *output_dims]
        """
        expected_timesteps = int(np.sum(
            (input_packets & 0x7) == IOManager.AcceleratorPacket.TYPE_TIMESTEP
        ))

        self.start_time = time.time()
        output_packets = self.sendreceive(input_packets, expected_timesteps)

        return self.io_manager.decode_output_packets_dual_numpy(output_packets)

    def infer_output_packets(self, input_packets: np.ndarray) -> np.ndarray:
        """Run inference from pre-encoded input packets and return raw output packets.

        This method returns the exact AXI output packet words (uint32) from hardware,
        with no decode, sign conversion, or quantized interpretation.

        Args:
            input_packets: uint32 numpy array of packed AXI input packets

        Returns:
            uint32 numpy array of raw AXI output packets
        """
        expected_timesteps = int(np.sum(
            (input_packets & 0x7) == IOManager.AcceleratorPacket.TYPE_TIMESTEP
        ))

        self.start_time = time.time()
        return self.sendreceive(input_packets, expected_timesteps)

    def infer_output_packet_ints(self, input_packets: np.ndarray) -> list[int]:
        """Run inference and return raw output packets as Python ints."""
        output_packets = self.infer_output_packets(input_packets)
        return [int(x) for x in output_packets]

    def infer_with_raw(self, input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference from dense input and return dequantized + raw outputs."""
        input_packets = self.io_manager.create_input_packets_numpy(input_array)
        return self.infer_from_packets_with_raw(input_packets)

    def close(self):
        """Shut down persistent worker threads."""
        self._shutdown.set()
        self._send_thread.join(timeout=3.0)
        self._recv_thread.join(timeout=3.0)

    def get_state(self) -> dict:
        """
        Read AXI-Lite management counters from AcceleratorAXI.

        Returns:
            Dictionary with cumulative timestep counters.
        """
        input_timesteps = int(self.accel_mmio.read(self.REG_INPUT_TS))
        output_timesteps = int(self.accel_mmio.read(self.REG_OUTPUT_TS))
        return {
            "input_timesteps": input_timesteps,
            "output_timesteps": output_timesteps,
        }

    def s2mm_status(self):
        """
        Read S2MM (receive) channel status and decode all flags.

        Returns:
            dict with all status flags and transfer info
        """
        # S2MM (Slave to Memory Mapped - Receive) registers
        S2MM_STATUS = 0x34  # Status register
        S2MM_LENGTH = 0x58  # Length register (bytes transferred)

        status = self.dma.mmio.read(S2MM_STATUS)
        length_bytes = self.dma.mmio.read(S2MM_LENGTH)

        return {
            "raw": f"{status:#010x}",
            "halted": bool(status & (1 << 0)),
            "idle": bool(status & (1 << 1)),
            "sg_included": bool(status & (1 << 3)),
            "dma_int_err": bool(status & (1 << 4)),
            "dma_slv_err": bool(status & (1 << 5)),
            "dma_dec_err": bool(status & (1 << 6)),
            "sg_int_err": bool(status & (1 << 8)),
            "sg_slv_err": bool(status & (1 << 9)),
            "sg_dec_err": bool(status & (1 << 10)),
            "ioc_irq": bool(status & (1 << 12)),
            "dly_irq": bool(status & (1 << 13)),
            "err_irq": bool(status & (1 << 14)),
            "length_bytes": length_bytes,
            "length_packets": length_bytes // 4,
        }


    def mm2s_status(self):
        """
        Read MM2S (send) channel status and decode all flags.

        Returns:
            dict with all status flags and transfer info
        """
        # MM2S (Memory Mapped to Stream - Send) registers
        MM2S_STATUS = 0x04  # Status register
        MM2S_LENGTH = 0x28  # Length register (bytes to transfer)

        status = self.dma.mmio.read(MM2S_STATUS)
        length_bytes = self.dma.mmio.read(MM2S_LENGTH)

        return {
            "raw": f"{status:#010x}",
            "halted": bool(status & (1 << 0)),
            "idle": bool(status & (1 << 1)),
            "sg_included": bool(status & (1 << 3)),
            "dma_int_err": bool(status & (1 << 4)),
            "dma_slv_err": bool(status & (1 << 5)),
            "dma_dec_err": bool(status & (1 << 6)),
            "sg_int_err": bool(status & (1 << 8)),
            "sg_slv_err": bool(status & (1 << 9)),
            "sg_dec_err": bool(status & (1 << 10)),
            "ioc_irq": bool(status & (1 << 12)),
            "dly_irq": bool(status & (1 << 13)),
            "err_irq": bool(status & (1 << 14)),
            "length_bytes": length_bytes,
            "length_packets": length_bytes // 4,
        }

def reload_modules():
    importlib.reload(IOManager)
