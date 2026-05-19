package NIR2FPGA

import spinal.core._
import spinal.core.sim._
import spinal.lib._

case class Activations(c: Activations.Config) extends Bundle {
  require(c.shape.nonEmpty, s"Shape cannot be empty: $c.shape")
  require(c.shape.forall(_ >= 1), s"All shape dimensions must be >= 1: $c.shape")
  require(c.shape.last >= c.width, s"Last dimension (${c.shape(0)}) should be greater than width $c.width")
  require(
    c.shape.last % c.width == 0,
    s"Last dimension (length: ${c.shape(0)}) should be divisible by width ($c.width)"
  )

  val coords = Vec(c.shapeWithWidth.map(d => UInt(log2Up(d) bits)))
  val value  = c.mkValue

  def flattenedAddress: UInt = coords
    .zip(c.shapeWithWidth.scanRight(1)(_ * _).tail)
    .map { case (c, s) => c * s }
    .reduce(_ + _)
    .resize(log2Up(c.linearSize) bits)

}

object Activations {

  case class Config(quant: QuantizationConfig, shape: Seq[Int], width: Int) {
    val shapeWithWidth = shape.init :+ (shape.last / width)

    val linearSize = shapeWithWidth.reduce(_ * _)

    def mkValue  = Vec(AFix(quant.qformat), width)
    def mkStream = Stream(Fragment(Activations(this)))
  }

}
