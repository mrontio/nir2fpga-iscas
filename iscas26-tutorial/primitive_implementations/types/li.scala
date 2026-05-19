package NIR2FPGA.Primitives

import spinal.core._
import spinal.lib._
import nir._

import NIR2FPGA.ConfigJSON
import NIR2FPGA.Activations
import NIR2FPGA.Primitives.Implementations.Neuron

final case class LIHW(
  id: String,
  params: LIParams,
  config: ConfigJSON
) extends PrimitiveHW[LIParams] {

  /* alpha = 1 - 2/tau */
  /* v[t] = alpha * v[t-1] + input[t] */

  def makeHardware(inputAct: Activations): (Stream[Fragment[Activations]], Stream[Fragment[Activations]]) = {
    // For now, assert that v_leak is zero
    val vLeakValue = NodeHelper.extractScalar(params.v_leak)
    require(vLeakValue == 0.0, s"LIParams: v_leak must be zero for now, got ${vLeakValue}")

    val liconfig = Neuron.Config(
      input = inputAct.c,
      quants = config.quantizations(id),
      tau = Some(NodeHelper.extractScalar(params.tau))
    )

    val neuron = Neuron(liconfig).setName("li")
    (neuron.input, neuron.output)
  }

}
