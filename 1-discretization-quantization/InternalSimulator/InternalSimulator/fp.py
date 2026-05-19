from __future__ import annotations

from typing import Union

import torch


class FP:
    # TODO: define as high power of 2 and low power of 2
    # e.g. "We want to represent from 1/2 to 1/8"
    FRACTIONAL_BITS = 24
    INTEGER_BITS = 8
    TOTAL_BITS = FRACTIONAL_BITS + INTEGER_BITS
    MAX_VALUE = (1 << (INTEGER_BITS + FRACTIONAL_BITS - 1)) - 1
    MIN_VALUE = -(1 << (INTEGER_BITS + FRACTIONAL_BITS - 1))

    def __init__(self, value: Union[float, int, FP]) -> None:
        if isinstance(value, float):
            self.raw = int(round(value * (1 << self.FRACTIONAL_BITS)))
        elif isinstance(value, int):
            self.raw = value << self.FRACTIONAL_BITS
        elif isinstance(value, FP):
            self.raw = value.raw
        else:
            raise TypeError("Unsupported type for FP initialization")
        self._check_overflow()

    @classmethod
    def from_raw(cls, raw_value: int) -> FP:
        """
        Create an FP instance by reinterpreting a raw integer as a fixed-point value,
        without shifting. This is known as a reinterpret cast in low-level languages.
        """
        obj = cls(0)
        obj.raw = raw_value
        obj._check_overflow()
        return obj

    def _check_overflow(self) -> None:
        if self.raw > self.MAX_VALUE:
            self.raw = self.MAX_VALUE
        elif self.raw < self.MIN_VALUE:
            self.raw = self.MIN_VALUE

    def __add__(self, other: object) -> FP:
        if not isinstance(other, FP):
            return NotImplemented
        result = FP(0)
        result.raw = self.raw + other.raw
        result._check_overflow()
        return result

    def __sub__(self, other: object) -> FP:
        if not isinstance(other, FP):
            return NotImplemented
        result = FP(0)
        result.raw = self.raw - other.raw
        result._check_overflow()
        return result

    def __mul__(self, other: object) -> FP:
        if not isinstance(other, FP):
            return NotImplemented
        result = FP(0)
        product = (self.raw * other.raw) >> self.FRACTIONAL_BITS
        result.raw = product
        result._check_overflow()
        return result

    def __truediv__(self, other: object) -> FP:
        if not isinstance(other, FP):
            return NotImplemented
        result = FP(0)
        quotient = (self.raw << self.FRACTIONAL_BITS) // other.raw
        result.raw = quotient
        result._check_overflow()
        return result

    def __gt__(self, other: FP) -> bool:
        return self.raw > other.raw

    def to_float(self) -> float:
        return self.raw / float(1 << self.FRACTIONAL_BITS)

    def __repr__(self) -> str:
        return f"Q{FP.INTEGER_BITS}_{FP.FRACTIONAL_BITS}({self.to_float()})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FP):
            return self.raw == other.raw
        return False


class FPTensor:
    """
    Fixed-point tensor using integer storage, matching hardware arithmetic exactly.

    This class stores values as raw integers and performs all operations in fixed-point,
    avoiding the float-then-floor pattern that causes simulation/hardware mismatch.
    """

    def __init__(self, values: torch.Tensor, int_bits: int, frac_bits: int, signed: bool = True):
        """
        Create FPTensor from float tensor.

        Args:
            values: Float tensor to convert to fixed-point
            int_bits: Number of integer bits (including sign bit for signed)
            frac_bits: Number of fractional bits
        """
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.signed = signed
        self.total_bits = int_bits + frac_bits
        self.scale = 1 << frac_bits
        if self.signed:
            self.max_val = (1 << (self.total_bits - 1)) - 1
            self.min_val = -(1 << (self.total_bits - 1))
        else:
            self.max_val = (1 << self.total_bits) - 1
            self.min_val = 0

        # Store as integer tensor (the raw fixed-point representation)
        # Use round() to match hardware rounding behavior
        self.raw = torch.round(values * self.scale).long()
        self._clamp()

    @classmethod
    def from_raw(cls, raw: torch.Tensor, int_bits: int, frac_bits: int, signed: bool = True) -> "FPTensor":
        """
        Create FPTensor from raw integer representation (no scaling).

        This is the equivalent of a reinterpret cast - the raw integers
        are used directly as the fixed-point representation.
        """
        obj = cls.__new__(cls)
        obj.int_bits = int_bits
        obj.frac_bits = frac_bits
        obj.signed = signed
        obj.total_bits = int_bits + frac_bits
        obj.scale = 1 << frac_bits
        if obj.signed:
            obj.max_val = (1 << (obj.total_bits - 1)) - 1
            obj.min_val = -(1 << (obj.total_bits - 1))
        else:
            obj.max_val = (1 << obj.total_bits) - 1
            obj.min_val = 0
        obj.raw = raw.long()
        obj._clamp()
        return obj

    def _clamp(self):
        """Clamp raw values to valid range (saturation arithmetic)."""
        self.raw = torch.clamp(self.raw, self.min_val, self.max_val)

    def _is_fp_tensor(self, other: object) -> bool:
        """Check if other is an FPTensor (handles module reload issues)."""
        return hasattr(other, 'raw') and hasattr(other, 'frac_bits') and hasattr(other, 'int_bits')

    def __add__(self, other: "FPTensor") -> "FPTensor":
        """Fixed-point addition."""
        if not self._is_fp_tensor(other):
            return NotImplemented
        # For addition, formats should match (same frac_bits at minimum)
        assert self.frac_bits == other.frac_bits, (
            f"Mismatched frac_bits: {self.frac_bits} vs {other.frac_bits}"
        )
        result_raw = self.raw + other.raw
        return FPTensor.from_raw(result_raw, self.int_bits, self.frac_bits, self.signed)

    def __sub__(self, other: "FPTensor") -> "FPTensor":
        """Fixed-point subtraction."""
        if not self._is_fp_tensor(other):
            return NotImplemented
        assert self.frac_bits == other.frac_bits, (
            f"Mismatched frac_bits: {self.frac_bits} vs {other.frac_bits}"
        )
        result_raw = self.raw - other.raw
        return FPTensor.from_raw(result_raw, self.int_bits, self.frac_bits, self.signed)

    def __mul__(self, other: "FPTensor") -> "FPTensor":
        """
        Fixed-point multiplication.

        In hardware: (a * b) >> frac_bits
        This matches the SpinalHDL AFix multiplication behavior.
        """
        if not self._is_fp_tensor(other):
            return NotImplemented
        # Multiply raw values and shift right by frac_bits
        # Note: this truncates (rounds toward zero), matching hardware behavior
        product = (self.raw * other.raw) >> self.frac_bits
        return FPTensor.from_raw(product, self.int_bits, self.frac_bits, self.signed)

    def __ge__(self, other: "FPTensor") -> torch.Tensor:
        """Greater-than-or-equal comparison, returns boolean tensor."""
        if not self._is_fp_tensor(other):
            return NotImplemented
        return self.raw >= other.raw

    def __gt__(self, other: "FPTensor") -> torch.Tensor:
        """Greater-than comparison, returns boolean tensor."""
        if not self._is_fp_tensor(other):
            return NotImplemented
        return self.raw > other.raw

    def __le__(self, other: "FPTensor") -> torch.Tensor:
        """Less-than-or-equal comparison, returns boolean tensor."""
        if not self._is_fp_tensor(other):
            return NotImplemented
        return self.raw <= other.raw

    def __lt__(self, other: "FPTensor") -> torch.Tensor:
        """Less-than comparison, returns boolean tensor."""
        if not self._is_fp_tensor(other):
            return NotImplemented
        return self.raw < other.raw

    def to_float(self) -> torch.Tensor:
        """Convert back to float tensor."""
        return self.raw.float() / self.scale

    def clone(self) -> "FPTensor":
        """Create a copy of this FPTensor."""
        return FPTensor.from_raw(self.raw.clone(), self.int_bits, self.frac_bits, self.signed)

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the underlying tensor."""
        return self.raw.shape

    def __repr__(self) -> str:
        prefix = "SQ" if self.signed else "UQ"
        return f"FPTensor({prefix}{self.int_bits}.{self.frac_bits}, shape={self.shape})"

    @staticmethod
    def where(condition: torch.Tensor, x: "FPTensor", y: "FPTensor") -> "FPTensor":
        """
        Element-wise selection based on condition.

        Like torch.where but for FPTensor.
        """
        assert x.frac_bits == y.frac_bits and x.int_bits == y.int_bits
        result_raw = torch.where(condition, x.raw, y.raw)
        return FPTensor.from_raw(result_raw, x.int_bits, x.frac_bits, x.signed)
