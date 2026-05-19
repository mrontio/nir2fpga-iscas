package NIR2FPGA.Primitives.Implementations

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.fsm._
import NIR2FPGA._

/**
 * Affine layer: y_j = Σ_i ( w[j][i] · x_i ) + bias_j, computed per timestep.
 *
 * Inputs stream in one neuron at a time; the fragment whose `last` flag is set
 * marks the end of a timestep. Products are accumulated into one register per
 * output neuron, then the N_out results are serialised out (width = 1).
 */
case class Affine(c: Affine.Config) extends Component {
  require(c.input.width == 1, "Affine (tutorial): input width must be 1")
  require(c.output.width == 1, "Affine (tutorial): output width must be 1")
  require(c.input.shape.length == 1, "Affine: input shape must be 1-D")
  require(c.output.shape.length == 1, "Affine: output shape must be 1-D")

  val input  = slave Stream (Fragment(Activations(c.input)))
  val output = (master Stream (Fragment(Activations(c.output)))).simPublic()

  val nIn    = c.input.shape.last
  val nOut   = c.output.shape.last
  val outFmt = c.output.quant.qformat

  // Constant weight matrix: weight(j) is a hardware Vec indexable by input coord.
  val weightData: List[List[Double]] =
    c.weights.w.toList.asInstanceOf[List[List[Double]]]

  val weight = Vec(weightData.map { row =>
    Vec(row.map(v => AF(c.weights.quantizedLiteral(v), c.weights.quant.qformat)))
  })

  val bias = Vec(c.bias.map(v => AF(v, outFmt)))

  // One wide accumulator register per output neuron.
  val acc         = Vec(Reg(AFix(c.accumFormat)) init AF(0.0, c.accumFormat), nOut)
  val sendCounter = Reg(UInt(log2Up(nOut + 1) bits)) init 0

  // Safe indexing for the degenerate size-1 case (0-bit coord / Vec of 1).
  def inIdx(addr: UInt): UInt = if (nIn == 1) U(0) else addr.resized
  def outIdx(idx: UInt): UInt = if (nOut == 1) U(0) else idx.resized

  // Defaults
  input.ready                       := False
  output.valid                      := False
  output.payload.last               := False
  output.payload.fragment.coords(0) := 0
  output.payload.fragment.value(0)  := AF(0.0, outFmt)

  val fsm = new StateMachine {
    val ACCUMULATE = new State with EntryPoint
    val SEND       = new State

    ACCUMULATE.whenIsActive {
      input.ready := True
      when(input.fire) {
        val i = inIdx(input.payload.fragment.flattenedAddress)
        val x = input.payload.fragment.value(0)
        for (j <- 0 until nOut)
          acc(j) := (acc(j) + (weight(j)(i) * x))
            .fixTo(c.accumFormat, RoundType.FLOOR)
        when(input.payload.last) {
          sendCounter := 0
          goto(SEND)
        }
      }
    }

    SEND.whenIsActive {
      val j = outIdx(sendCounter)
      output.valid                      := True
      output.payload.last               := sendCounter === U(nOut - 1)
      output.payload.fragment.coords(0) := sendCounter.resized
      output.payload.fragment.value(0)  :=
        (acc(j) + bias(j)).fixTo(outFmt, RoundType.FLOOR)
      when(output.fire) {
        acc(j) := AF(0.0, c.accumFormat) // clear for the next timestep
        when(sendCounter === U(nOut - 1)) {
          goto(ACCUMULATE)
        }.otherwise {
          sendCounter := sendCounter + 1
        }
      }
    }
  }

}

object Affine {

  case class Config(
    input: Activations.Config,
    output: Activations.Config,
    accumFormat: QFormat,
    weights: Weights,
    bias: Seq[Double] // already quantised into the output format
  )

}
