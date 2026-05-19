package NIR2FPGA

import spinal.core._
import spinal.lib._
import nir._
import nir.tensor.Tensor
import NIR2FPGA.Primitives._

case class AcceleratorConfig(
  reduction: Boolean = false,
  spikeGating: Boolean = true,
  macWidth: Int = 1
)

object AcceleratorConfig {
  def default: AcceleratorConfig = AcceleratorConfig()
}

// Assuming nir graph specifies incoming edges
case class Accelerator(
  nirGraph: NIRGraph,
  config: ConfigJSON,
  accelConfig: AcceleratorConfig = AcceleratorConfig.default
) extends Component {
  // Used to verify the VCD version
  val configTimestamp = Reg(Bits(32 bits)) init B(config.timestamp)
  configTimestamp.addAttribute("keep")

  val inputParams  = nirGraph.input.params.asInstanceOf[InputParams]
  val outputParams = nirGraph.output.params.asInstanceOf[OutputParams]

  // Convert NIR tensor shape to Seq[Int] for Activations
  val inputShape: Seq[Int]  = inputParams.shape.map(_.toInt).toFlatList
  val outputShape: Seq[Int] = outputParams.shape.map(_.toInt).toFlatList

  val inputConfig  = Activations.Config(config.quantizations("input")("output"), inputShape, 1)
  val outputConfig = Activations.Config(config.quantizations("output")("input"), outputShape, 1)

  val io = new Bundle {
    val input  = slave Stream (Fragment(Activations(inputConfig)))
    val output = master Stream (Fragment(Activations(outputConfig)))
  }

  val graphReduced = if (accelConfig.reduction) NIRGraph.reduce(nirGraph) else nirGraph
  val resolved     = resolve(graphReduced.output)

  // Downsize to width=1 at the output boundary if internal layers used wider width
  if (accelConfig.macWidth > 1 && resolved.payload.fragment.c.width > 1) {
    val downsizer = Downsizer(resolved.payload.fragment.c, outputConfig)
    downsizer.i << resolved
    io.output << downsizer.o
  } else {
    io.output << resolved
  }

  // Maps NIR graph to hardware
  def resolve(node: NIRNode): Stream[Fragment[Activations]] =
    node.params match {
      case InputParams(_) =>
        io.input;
      case OutputParams(_) =>
        node.previous.size match {
          case 1 => resolve(node.previous.toSeq(0))
          case n => throw new Exception(s"${n} incoming edges not yet supported.")
        }
      case params =>
        val incoming = node.previous.map(resolve(_)).toSeq
        val input    = incoming match {
          case Seq(single) => single
          case n           => throw new Exception(s"${n} incoming edges not yet supported.")
        }

        // Pipeline/register the input before passing to hardware node
        val pipelined_input = input.m2sPipe()

        // Apply spike gating before compute-intensive nodes (Affine, Conv)
        val shouldGate = accelConfig.spikeGating && (params match {
          case _: AffineParams    => true
          case _: AffineLIFParams => true
          case _                  => false
        })

        val gated_input = if (shouldGate) {
          val gate = SpikeGate(pipelined_input.payload.fragment.c)
          gate.i << pipelined_input
          gate.o
        } else {
          pipelined_input
        }

        val (hw_input, hw_output) =
          PrimitiveHW
            .create(node.id, params, config, accelConfig)
            .makeHardware(gated_input.payload.fragment)

        hw_input << gated_input
        hw_output
    }

}
