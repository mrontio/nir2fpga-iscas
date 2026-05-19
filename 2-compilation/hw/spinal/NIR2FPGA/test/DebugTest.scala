package NIR2FPGA.test

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import spinal.core._
import spinal.core.sim._
import spinal.lib.sim.StreamDriver
import spinal.lib.sim.StreamMonitor
import scala.collection.mutable.Queue
import scala.collection.mutable.ArrayBuffer

import scala.io.Source
import java.io.File

import nir._
import NIR2FPGA.ConfigJSON
import tensor._

/**
 * Test used for one-off tests for development.
 */
class DebugTest extends AnyFunSuite with Matchers {
  test("Main") {
    println("Incomplete test btw")

    val jsonFile   = "../train/tests/lif/model.json"
    val source     = Source.fromFile(jsonFile)
    val jsonString =
      try source.mkString
      finally source.close()
    val configJson = ConfigJSON.fromJson(jsonString)

    println(configJson.input.shape)
    println(configJson.quantizations)
  }
}
