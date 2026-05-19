import torch
from enum import IntEnum, Enum
from typing import Dict, Optional, Set

class RecordingType(IntEnum):
    SOURCE = 0
    INTERNAL = 1
    QUANTIZED = 2
    BEHAVIOURAL = 3
    HARDWARE = 4

class RecordingCompleteness(Enum):
    FULL = "full"
    PARTIAL = "partial"


class Recording:
    REQUIREMENTS: Dict[str, Set[str]] = {
        "IAFSqueeze": {"v_mem"},
        "LIFSqueeze": {"v_mem"},
    }

    def __init__(
        self,
        recording_type: RecordingType,
        data: Dict[str, Dict[str, torch.Tensor]],
        layer_types: Optional[Dict[str, str]] = None,
        internal_recording: Optional["Recording"] = None,
    ):
        self.recording_type = recording_type
        self.data = data

        if layer_types is not None:
            self._validate_requirements(layer_types)

        self._validate_structure()
        self.completeness = self._classify(internal_recording)

    def _validate_requirements(self, layer_types: Dict[str, str]) -> None:
        for layer_id, layer_rec in self.data.items():
            if layer_id not in layer_types:
                continue
            layer_type = layer_types[layer_id]
            required_params = Recording.REQUIREMENTS.get(layer_type, set())
            provided_params = set(layer_rec.keys())
            missing = required_params - provided_params
            if missing:
                raise ValueError(
                    f"Recording for layer '{layer_id}' ({layer_type}) in {self.recording_type.value} "
                    f"is missing required parameters: {missing}"
                )

    INTERNAL_TYPES = {RecordingType.INTERNAL, RecordingType.QUANTIZED}

    def _validate_structure(self) -> None:
        nodes = set(self.data.keys())

        if self.recording_type in self.INTERNAL_TYPES:
            # Internal simulator recordings must have a graph-level "output" node
            if "output" not in nodes:
                raise ValueError(
                    f"Recording '{self.recording_type.value}' must have 'output' node"
                )
            if "input" not in self.data["output"]:
                raise ValueError(
                    f"Recording '{self.recording_type.value}' must have 'output' node with 'input' parameter."
                )

    def _classify(self, internal_recording: Optional["Recording"]) -> RecordingCompleteness:
        if internal_recording is None:
            return RecordingCompleteness.FULL

        internal_data = internal_recording.data
        internal_ids = [k for k in internal_data.keys() if k not in ("input", "output")]

        for layer_id in internal_ids:
            if layer_id not in self.data:
                return RecordingCompleteness.PARTIAL
            for param, tensor in self.data[layer_id].items():
                if param in internal_data.get(layer_id, {}):
                    internal_shape = internal_data[layer_id][param].shape
                    rec_shape = tensor.shape
                    if rec_shape != internal_shape:
                        return RecordingCompleteness.PARTIAL

        return RecordingCompleteness.FULL

    def layers(self):
        return self.keys()

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key: str) -> Dict[str, torch.Tensor]:
        return self.data[key]

    def validate_for_plotting(self, neuron_node_id: str) -> None:
        if neuron_node_id in self.data and self.recording_type != RecordingType.HARDWARE:
            param_keys = self.data[neuron_node_id].keys()
            if "v_mem" not in param_keys:
                raise ValueError(
                    f"Recording '{self.recording_type.value}' at indicated neuron node "
                    f"'{neuron_node_id}' must have 'v_mem' parameter."
                )
