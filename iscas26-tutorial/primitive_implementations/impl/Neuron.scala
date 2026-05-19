package NIR2FPGA.Primitives.Implementations

import spinal.core._
import spinal.core.sim._
import spinal.lib._

import NIR2FPGA.Activations
import NIR2FPGA.QuantizationConfig

case class Neuron(c: Neuron.Config) extends Component {
  val input        = slave Stream (Fragment(Activations(c.input)))
  val outputConfig = Activations.Config(c.quants("output"), c.input.shape, c.input.width)
  val output       = master Stream Fragment(Activations(outputConfig)).simPublic()

  val mem = Mem(Vec(AFix(c.quants("v_mem").qformat), c.input.width), c.input.linearSize)
    .initBigInt(List.fill(c.input.linearSize)(BigInt(0)))

  // Simulation probe for v_mem. Each probe word wraps its membrane values in a
  // `value` field so the VCD signal name is `v_mem_<word>_value_<lane>`, which
  // is what VCDMapping.neuron_v_mem_factory searches for.
  val debug_mem_probe = Vec.fill(scala.math.min(c.input.linearSize, 4))(new Bundle {
    val value = Vec(AFix(c.quants("v_mem").qformat), c.input.width)
  })

  debug_mem_probe.simPublic()
  debug_mem_probe.setName("v_mem")

  for (i <- debug_mem_probe.indices)
    debug_mem_probe(i).value := mem.readAsync(U(i, log2Up(c.input.linearSize) bits))

  val addr = input.payload.fragment.flattenedAddress

  val memReadWithPayload = mem.streamReadSync(
    input.translateWith(addr),
    input.payload
  )

  // Leak factor: alpha = 1 - 2/tau. With tau = None, alpha = 1 (no leak),
  // which makes this neuron behave as a plain integrator (I).
  val alpha = c.tau match {
    case Some(tau) => 1.0 - 2.0 / tau
    case None      => 1.0
  }

  val alphaFactor = Vec(AF(alpha, c.quants("v_mem").qformat), c.input.width)

  // Leak: alpha * v[t-1]
  val leaked = memReadWithPayload.map { p =>
    val state = p.value
    val input = p.linked

    val leakedState = Vec(
      state
        .zip(alphaFactor)
        .map { case (v, alpha) =>
          (alpha * v)
            .fixTo(c.quants("v_mem").qformat, RoundType.CEIL)
        }
    )

    TupleBundle(leakedState, input)
  }

  // Integrate: v[t] = alpha * v[t-1] + input[t]
  val integrated = leaked.map { p =>
    val leakedState = p._1
    val input       = p._2

    val integratedState = Vec(leakedState.zip(input.fragment.value).map { case (v, inp) =>
      (v + inp)
        .fixTo(c.quants("v_mem").qformat, RoundType.CEIL)
    })

    TupleBundle(integratedState, input)
  }

  // Write updated membrane back
  mem.write(
    address = integrated.payload._2.fragment.flattenedAddress,
    data = integrated.payload._1,
    enable = integrated.fire
  )

  // --- Drive output ---
  integrated.translateInto(output) { case (out, from) =>
    val state = from._1
    val input = from._2
    out.last            := input.last
    out.fragment.coords := input.fragment.coords
    out.fragment.value  := Vec(
      state.map(_.fixTo(c.quants("output").qformat, RoundType.FLOOR))
    )
  }
}

object Neuron {

  case class Config(
    input: Activations.Config,
    quants: Map[String, QuantizationConfig],
    tau: Option[Double] = None
  )

}
