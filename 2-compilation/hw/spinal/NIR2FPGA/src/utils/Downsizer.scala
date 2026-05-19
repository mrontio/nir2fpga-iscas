package NIR2FPGA

import NIR2FPGA._
import spinal.core._
import spinal.lib._

case class Downsizer(ic: Activations.Config, oc: Activations.Config, swapLastAndFirstDim: Boolean = false)
    extends Component {
  require(ic.width % oc.width == 0, "Input width should be divisible by output width")
  require(
    ic.width > oc.width || (swapLastAndFirstDim && ic.width == oc.width),
    s"Output width should be smaller than input width (or equal when swapping), got ic.width=${ic.width} oc.width=${oc.width}"
  )

  if (swapLastAndFirstDim) {
    require(ic.shape.length >= 2, "Cannot swap dimensions when there is only 1 dimension")
    val expandedLastDim = ic.shapeWithWidth.last * (ic.width / oc.width)
    require(
      oc.shapeWithWidth.head == expandedLastDim,
      s"When swapping, output first dim (${oc.shapeWithWidth.head}) should equal expanded input last dim ($expandedLastDim)"
    )
    require(
      oc.shapeWithWidth.last == ic.shapeWithWidth.head,
      s"When swapping, output last dim (${oc.shapeWithWidth.last}) should equal input first dim (${ic.shapeWithWidth.head})"
    )
    require(
      oc.shapeWithWidth.tail.init == ic.shapeWithWidth.tail.init,
      s"When swapping, middle dimensions should match: output ${oc.shapeWithWidth.tail.init} vs input ${ic.shapeWithWidth.tail.init}"
    )
  }

  val i = slave(ic.mkStream)
  val o = master(oc.mkStream)

  StreamTransactionExtender(
    i,
    o,
    count = (ic.width / oc.width) - 1
  ) { (counter, vec, last) =>
    val out = cloneOf(o.payload)
    out.last := vec.last && last
    val grouped = Vec(vec.fragment.value.grouped(oc.width).toSeq.map(Vec(_)))

    out.fragment.value := grouped(counter.resized)

    val newLastCoord = (vec.fragment.coords.last * U(ic.width / oc.width) + counter).resized
    val outCoords    = if (swapLastAndFirstDim) {
      Vec(newLastCoord +: vec.fragment.coords.tail.init :+ vec.fragment.coords.head)
    } else {
      Vec(vec.fragment.coords.init :+ newLastCoord)
    }
    out.fragment.coords := outCoords

    out
  }
}
