package NIR2FPGA

import spinal.core._
import io.circe.{Decoder, DecodingFailure, HCursor}
import io.circe.parser.decode
import scala.math._

import nir.tensor.Tensor

class QuantizationConfig(
  val qformat: QFormat
)

class ConfigJSON(
  val input: Tensor[Double],
  val quantizations: Map[String, Map[String, QuantizationConfig]],
  val timesteps: Int,
  val timestamp: Int,
  val inputPackets: Option[List[Long]] = None
)

object ConfigJSON {
  import Tensor.doubleDecoder

  implicit val quantizationConfigDecoder: Decoder[QuantizationConfig] = Decoder.instance { cursor =>
    for {
      minValue  <- cursor.get[Int]("min_value")
      maxValue  <- cursor.get[Int]("max_value")
      exponent  <- cursor.get[Int]("exp")
      bits      <- cursor.get[Int]("bits")
      frac_bits <- cursor.get[Int]("frac_bits")
      int_bits  <- cursor.get[Int]("int_bits")
      signed    <- cursor.get[Boolean]("signed")
    } yield {
      val qformat = QFormat(bits, frac_bits, signed)
      new QuantizationConfig(qformat)
    }
  }

  private implicit val longDecoder: Decoder[Long] = Decoder.decodeLong

  implicit val jsonConfigDecoder: Decoder[ConfigJSON] = Decoder.instance { cursor =>
    for {
      input         <- cursor.downField("input").as[Tensor[Double]]
      quantizations <- cursor.downField("quantizations").as[Map[String, Map[String, QuantizationConfig]]]
      timesteps     <- cursor.get[Int]("timesteps")
      timestamp     <- cursor.get[Int]("timestamp")
      inputPackets  <- cursor.get[Option[List[Long]]]("input_packets")
    } yield new ConfigJSON(input, quantizations, timesteps, timestamp, inputPackets)
  }

  def fromJson(json: String): ConfigJSON =
    decode[ConfigJSON](json) match {
      case Right(data) =>
        data

      case Left(error) =>
        println("❌ Failed to parse JSON:")
        println(s"   Error: ${error.getMessage()}")

        // Try to extract more context from the error
        error match {
          case DecodingFailure(message, path) =>
            println(s"   Path: ${path.mkString(".")}")
            println(s"   Message: $message")

          // No special hints currently
          case _ =>
            println(s"   Details: $error")
        }

        throw new RuntimeException("JSON parsing failed", error)
    }

  def fromFile(filePath: String): ConfigJSON = {
    val json = scala.io.Source.fromFile(filePath).mkString
    fromJson(json)
  }

}

class ThroughputConfig(
  val numSamples: Int,
  val timesteps: Int,
  val samples: List[List[Long]]
)

object ThroughputConfig {
  import io.circe.parser.decode

  private implicit val uint32Decoder: Decoder[Long] = Decoder.decodeLong.emap { value =>
    if (value < 0 || value > 0xffffffffL) Left(s"Value $value out of uint32 range")
    else Right(value)
  }

  private implicit val uint32ListDecoder: Decoder[List[Long]] = Decoder.decodeList(uint32Decoder)

  implicit val throughputDecoder: io.circe.Decoder[ThroughputConfig] = io.circe.Decoder.instance { c =>
    for {
      numSamples <- c.get[Int]("num_samples")
      timesteps  <- c.get[Int]("timesteps")
      samples    <- c.get[List[List[Long]]]("samples")
    } yield new ThroughputConfig(numSamples, timesteps, samples)
  }

  def fromJson(json: String): ThroughputConfig =
    decode[ThroughputConfig](json).fold(e => throw new RuntimeException(e.getMessage), identity)
}
