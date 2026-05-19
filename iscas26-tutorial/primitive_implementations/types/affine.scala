package NIR2FPGA.Primitives

import spinal.core._
import spinal.lib._
import nir._

import NIR2FPGA.ConfigJSON
import NIR2FPGA.Activations
import NIR2FPGA.Weights
import NIR2FPGA.Primitives.Implementations.Affine

final case class AffineHW(
  id: String,
  params: AffineParams,
  config: ConfigJSON
) extends PrimitiveHW[AffineParams] {

  /* y_j = Σ_i ( w[j][i] · x_i ) + bias_j */

  def makeHardware(inputAct: Activations): (Stream[Fragment[Activations]], Stream[Fragment[Activations]]) = {
    val quant   = config.quantizations(id)
    val weights = Weights(params.weight.map(_.toDouble), quant("weights"))

    // Serial (width 1) input/output streams: input shape [N_in], output [N_out].
    val inputConfig  = Activations.Config(quant("input"), weights.inputs, 1)
    val outputConfig = Activations.Config(quant("output"), weights.outputs, 1)

    // Accumulator widened by log2(N_in) integer bits so a sum of N_in products
    // cannot overflow the output format.
    val outFmt      = outputConfig.quant.qformat
    val nIn         = weights.inputs.last
    val accumFormat = QFormat(outFmt.width + log2Up(nIn), outFmt.fraction, outFmt.signed)

    // Quantise bias into the output fixed-point format (floor + saturate).
    val biasScale          = 1 << outFmt.fraction
    val (biasMin, biasMax) = if (outFmt.signed) {
      (-(1 << (outFmt.width - 1)), (1 << (outFmt.width - 1)) - 1)
    } else {
      (0, (1 << outFmt.width) - 1)
    }
    val bias = params.bias.toFlatList.map(_.toDouble).map { v =>
      val raw = scala.math.floor(v * biasScale).toInt.max(biasMin).min(biasMax)
      raw.toDouble / biasScale
    }

    val affine = Affine(Affine.Config(inputConfig, outputConfig, accumFormat, weights, bias))
      .setName("affine")
    (affine.input, affine.output)
  }

}
