package NIR2FPGA

import spinal.core._
import spinal.core.sim._
import spinal.lib._

import nir.tensor.Tensor

case class Weights(w: Tensor[Double], quant: QuantizationConfig) extends Bundle {
  require(w.shape.nonEmpty, s"Shape cannot be empty: $w.shape")
  require(w.rank == 2, s"Only weights of rank 2 supported, got: $w.shape")
  require(w.shape.forall(_ >= 1), s"All shape dimensions must be >= 1: $w.shape")

  def inputs: Seq[Int]  = List(w.shape(1))
  def outputs: Seq[Int] = List(w.shape(0))

  private def truncTowardZero(value: Double): Double =
    if (value >= 0.0) scala.math.floor(value) else scala.math.ceil(value)

  def quantizedLiteral(value: Double): Double = {
    val fracBits = quant.qformat.fraction
    val width    = quant.qformat.width
    val signed   = quant.qformat.signed
    val scale    = 1 << fracBits
    val minRaw   = if (signed) {
      -(1 << (width - 1))
    } else {
      0
    }
    val maxRaw = if (signed) {
      (1 << (width - 1)) - 1
    } else {
      (1 << width) - 1
    }

    val raw = truncTowardZero(value * scale).toInt.max(minRaw).min(maxRaw)
    raw.toDouble / scale.toDouble
  }

}
