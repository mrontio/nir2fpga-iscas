package NIR2FPGA.Primitives

import spinal.core._
import spinal.lib._
import nir._

import NIR2FPGA.ConfigJSON
import NIR2FPGA.Activations
import NIR2FPGA.Primitives.Implementations.Neuron

final case class IHW(
  id: String,
  params: IParams,
  config: ConfigJSON
) extends PrimitiveHW[IParams] {

  /* v[t] = v[t-1] + input[t] */

  def makeHardware(inputAct: Activations): (Stream[Fragment[Activations]], Stream[Fragment[Activations]]) = {
    val iconfig = Neuron.Config(
      input = inputAct.c,
      quants = config.quantizations(id)
    )

    val neuron = Neuron(iconfig).setName("i")
    (neuron.input, neuron.output)
  }

}
