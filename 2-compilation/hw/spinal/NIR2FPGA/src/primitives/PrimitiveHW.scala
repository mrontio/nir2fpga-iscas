package NIR2FPGA.Primitives

import spinal.core._
import spinal.lib._
import nir._
import nir.tensor.Tensor

import NIR2FPGA.Activations
import NIR2FPGA.AcceleratorConfig
import NIR2FPGA.ConfigJSON

trait PrimitiveHW[P <: NIRParams] {
  val params: P
  def makeHardware(inputAct: Activations): (Stream[Fragment[Activations]], Stream[Fragment[Activations]])
}

object PrimitiveHW {

  /** Compute the largest divisor of outputSize that is <= macWidth */
  def effectiveWidth(macWidth: Int, outputSize: Int): Int =
    (macWidth to 1 by -1).find(outputSize % _ == 0).getOrElse(1)

  def create(id: String, params: NIRParams, config: ConfigJSON, accelConfig: AcceleratorConfig): PrimitiveHW[_] =
    params match {
      case _ => throw new Exception(f"Not yet supported: ${params.getClass()}")
    }

}

object NodeHelper {

  def extractTensorValues(tensor: Tensor[Float]): List[Double] =
    tensor.map(_.toDouble).toFlatList

  def extractScalar(tensor: Tensor[Float]): Double = {
    val values = extractTensorValues(tensor)
    require(values.nonEmpty, s"Tensor must contain at least one value")
    values.head
  }

  def extractLongTensorValues(tensor: Tensor[Long]): List[Int] =
    tensor.map(_.toInt).toFlatList

}
