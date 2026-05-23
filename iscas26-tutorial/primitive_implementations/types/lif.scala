package NIR2FPGA.Primitives

import spinal.core._
import spinal.lib._
import nir._

import NIR2FPGA.ConfigJSON
import NIR2FPGA.Activations
import NIR2FPGA.Primitives.Implementations.Neuron

final case class LIFHW(
  id: String,
  params: LIFParams,
  config: ConfigJSON
) extends PrimitiveHW[LIFParams] {

  /* alpha = 1 - 2/tau */
  /* v[t] = alpha * v[t-1] + (1 - alpha) * input[t] */
  /* spike when v[t] >= v_threshold, then reset the membrane to 0 */

  def makeHardware(inputAct: Activations): (Stream[Fragment[Activations]], Stream[Fragment[Activations]]) = {
    // For now, assert that v_leak is zero
    val vLeakValue = NodeHelper.extractScalar(params.v_leak)
    require(vLeakValue == 0.0, s"LIFParams: v_leak must be zero for now, got ${vLeakValue}")

    val lifconfig = Neuron.Config(
      input = inputAct.c,
      quants = config.quantizations(id),
      tau = Some(NodeHelper.extractScalar(params.tau)),
      v_reset = Some(NodeHelper.extractScalar(params.v_reset)),
      v_threshold = Some(NodeHelper.extractScalar(params.v_threshold))
    )

    val neuron = Neuron(lifconfig).setName("lif")
    (neuron.input, neuron.output)
  }

}
