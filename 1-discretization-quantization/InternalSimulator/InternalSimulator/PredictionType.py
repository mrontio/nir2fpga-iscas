import operator
import torch

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List, Callable
from abc import ABC, abstractmethod

from InternalSimulator.Recording import Recording


@dataclass
class PredictionType(ABC):
    @abstractmethod
    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        pass

    @staticmethod
    def compare(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        comparison_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = operator.eq,
    ) -> float:
        n = min(predictions.shape[0], labels.shape[0])
        correct = comparison_function(predictions[:n], labels[:n])
        return correct.float().mean().item()  # type: ignore[return-value]


@dataclass
class LayerEqual(PredictionType):
    layer: str
    parameter: str
    atol: float = 0.0
    rtol: float = 0.0

    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        if self.layer not in recordings.keys():
            raise KeyError(f"Layer '{self.layer}' not found in recordings.")

        if self.parameter not in recordings[self.layer].keys():
            raise KeyError(f"Parameter '{self.parameter}' not found in recordings['{self.layer}'].")

        return PredictionType.compare(
            recordings.data[self.layer][self.parameter], labels,
            lambda a, b: torch.isclose(a, b, atol=self.atol, rtol=self.rtol),
        )

@dataclass
class OutputEqual(PredictionType):
    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        predicted = recordings.data["output"]["input"]
        return PredictionType.compare(predicted, labels)

@dataclass
class OutputSumMax(PredictionType):
    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        predicted = recordings.data["output"]["input"].sum(dim=1).argmax(dim=-1)
        return PredictionType.compare(predicted, labels)

@dataclass
class OutputAvgMax(PredictionType):
    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        _, predicted = torch.mean(recordings.data["output"]["input"], dim=1).max(dim=-1)
        return PredictionType.compare(predicted, labels)

@dataclass
class SoftMax(PredictionType):
    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        predictions = torch.nn.functional.softmax(recordings.data["output"]["input"].sum(dim=1), dim=-1).argmax(dim=-1)
        return PredictionType.compare(predictions, labels)

@dataclass
class Custom(PredictionType):
    prediction_fn: Callable[[Recording], torch.Tensor]
    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        predictions = self.prediction_fn(recordings)
        return PredictionType.compare(predictions, labels)

@dataclass
class Ignored(PredictionType):
    def __call__(self, recordings: Recording, labels: torch.Tensor) -> float:
        return False
