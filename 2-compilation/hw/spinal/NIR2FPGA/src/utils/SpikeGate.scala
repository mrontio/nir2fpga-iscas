package NIR2FPGA

import spinal.core._
import spinal.lib._
import NIR2FPGA._

case class SpikeGate(c: Activations.Config) extends Component {
  val i = slave Stream Fragment(Activations(c))
  val o = master Stream Fragment(Activations(c))

  val hasNonZero = i.payload.fragment.value.map(_.raw =/= 0).reduce(_ || _)
  o << i.takeWhen(hasNonZero || i.payload.last)
}
