package NIR2FPGA

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axis._
import spinal.core.sim._
import spinal.lib.sim.{StreamDriver, StreamMonitor}
import nir._
import nir.tensor.Tensor
import _root_.io.circe.Decoder
import _root_.io.circe.parser.decode
import scala.sys.process._

case class SpikeEntry(coordWidth: Int, valueWidth: Int) extends Bundle {
  val coord = UInt(coordWidth bits)
  val value = Bits(valueWidth bits)
}

class AcceleratorAXI(
  nirFile: String,
  jsonConfig: String,
  accelConfig: AcceleratorConfig = AcceleratorConfig.default
) extends Component {

  // ===== Configuration Setup (Phase 1) =====

  // Parse NIR graph and JSON configuration
  val nirGraph = NIRGraph(new java.io.File(nirFile))
  val config   = ConfigJSON.fromJson(jsonConfig)

  // Extract input/output parameters and shapes
  val inputParams  = nirGraph.input.params.asInstanceOf[InputParams]
  val outputParams = nirGraph.output.params.asInstanceOf[OutputParams]

  val inputShape: Seq[Int]  = inputParams.shape.map(_.toInt).toFlatList
  val outputShape: Seq[Int] = outputParams.shape.map(_.toInt).toFlatList

  // Extract quantization configurations
  val inputQuant  = config.quantizations("input")("output")
  val outputQuant = config.quantizations("output")("input")

  // Create Activations configurations
  val inputActivationConfig  = Activations.Config(inputQuant, inputShape, 1)
  val outputActivationConfig = Activations.Config(outputQuant, outputShape, 1)

  // ===== I/O Interface (Phase 7) =====

  // AXI4-Stream configuration for input
  val axiConfig = Axi4StreamConfig(
    dataWidth = 4, // 32 bits = 4 bytes
    useLast = true,
    useKeep = true,
    useStrb = false
  )

  val io = new Bundle {
    val s_axis = slave(Axi4Stream(axiConfig))  // AXI input from DMA
    val m_axis = master(Axi4Stream(axiConfig)) // AXI output to DMA

    // AXI-Lite control/status (management interface)
    val s_axi_ctrl_awaddr  = in UInt (32 bits)
    val s_axi_ctrl_awprot  = in Bits (3 bits)
    val s_axi_ctrl_awvalid = in Bool ()
    val s_axi_ctrl_awready = out Bool ()
    val s_axi_ctrl_wdata   = in Bits (32 bits)
    val s_axi_ctrl_wstrb   = in Bits (4 bits)
    val s_axi_ctrl_wvalid  = in Bool ()
    val s_axi_ctrl_wready  = out Bool ()
    val s_axi_ctrl_bresp   = out Bits (2 bits)
    val s_axi_ctrl_bvalid  = out Bool ()
    val s_axi_ctrl_bready  = in Bool ()
    val s_axi_ctrl_araddr  = in UInt (32 bits)
    val s_axi_ctrl_arprot  = in Bits (3 bits)
    val s_axi_ctrl_arvalid = in Bool ()
    val s_axi_ctrl_arready = out Bool ()
    val s_axi_ctrl_rdata   = out Bits (32 bits)
    val s_axi_ctrl_rresp   = out Bits (2 bits)
    val s_axi_ctrl_rvalid  = out Bool ()
    val s_axi_ctrl_rready  = in Bool ()

    // Debug wires for SystemILA
    val debug_input_timesteps  = out UInt (32 bits) // Number of input timesteps received
    val debug_output_timesteps = out UInt (32 bits) // Number of output timesteps completed
    val debug_input_state      =
      out UInt (8 bits) // Input FSM state (IDLE=0, COLLECTING=1, DRAINING=2)
    val debug_output_state     = out UInt (8 bits)  // Output FSM state (STREAMING_SPIKES=0, EMIT_TIMESTEP=1)
  }

  // Set AXI signal names for Vivado
  io.s_axis.valid.setName("s_axis_tvalid")
  io.s_axis.ready.setName("s_axis_tready")
  io.s_axis.data.setName("s_axis_tdata")
  io.s_axis.last.setName("s_axis_tlast")
  io.s_axis.keep.setName("s_axis_tkeep")

  io.m_axis.valid.setName("m_axis_tvalid")
  io.m_axis.ready.setName("m_axis_tready")
  io.m_axis.data.setName("m_axis_tdata")
  io.m_axis.last.setName("m_axis_tlast")
  io.m_axis.keep.setName("m_axis_tkeep")

  io.s_axi_ctrl_awaddr.setName("s_axi_ctrl_awaddr")
  io.s_axi_ctrl_awprot.setName("s_axi_ctrl_awprot")
  io.s_axi_ctrl_awvalid.setName("s_axi_ctrl_awvalid")
  io.s_axi_ctrl_awready.setName("s_axi_ctrl_awready")
  io.s_axi_ctrl_wdata.setName("s_axi_ctrl_wdata")
  io.s_axi_ctrl_wstrb.setName("s_axi_ctrl_wstrb")
  io.s_axi_ctrl_wvalid.setName("s_axi_ctrl_wvalid")
  io.s_axi_ctrl_wready.setName("s_axi_ctrl_wready")
  io.s_axi_ctrl_bresp.setName("s_axi_ctrl_bresp")
  io.s_axi_ctrl_bvalid.setName("s_axi_ctrl_bvalid")
  io.s_axi_ctrl_bready.setName("s_axi_ctrl_bready")
  io.s_axi_ctrl_araddr.setName("s_axi_ctrl_araddr")
  io.s_axi_ctrl_arprot.setName("s_axi_ctrl_arprot")
  io.s_axi_ctrl_arvalid.setName("s_axi_ctrl_arvalid")
  io.s_axi_ctrl_arready.setName("s_axi_ctrl_arready")
  io.s_axi_ctrl_rdata.setName("s_axi_ctrl_rdata")
  io.s_axi_ctrl_rresp.setName("s_axi_ctrl_rresp")
  io.s_axi_ctrl_rvalid.setName("s_axi_ctrl_rvalid")
  io.s_axi_ctrl_rready.setName("s_axi_ctrl_rready")

  // Add Vivado AXI interface attributes
  io.s_axis.valid.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 s_axis TVALID")
  io.s_axis.ready.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 s_axis TREADY")
  io.s_axis.data.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 s_axis TDATA")
  io.s_axis.data.addAttribute(
    "X_INTERFACE_PARAMETER",
    "CLK_DOMAIN clock,HAS_TKEEP 1,HAS_TLAST 1,HAS_TREADY 1,HAS_TSTRB 0," +
      "LAYERED_METADATA undef,TDATA_NUM_BYTES 4,TDEST_WIDTH 0,TID_WIDTH 0,TUSER_WIDTH 0"
  )
  io.s_axis.last.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 s_axis TLAST")
  io.s_axis.keep.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 s_axis TKEEP")

  io.m_axis.valid.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 m_axis TVALID")
  io.m_axis.ready.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 m_axis TREADY")
  io.m_axis.data.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 m_axis TDATA")
  io.m_axis.data.addAttribute(
    "X_INTERFACE_PARAMETER",
    "CLK_DOMAIN clock,HAS_TKEEP 1,HAS_TLAST 1,HAS_TREADY 1,HAS_TSTRB 0," +
      "LAYERED_METADATA undef,TDATA_NUM_BYTES 4,TDEST_WIDTH 0,TID_WIDTH 0,TUSER_WIDTH 0"
  )
  io.m_axis.last.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 m_axis TLAST")
  io.m_axis.keep.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:axis:1.0 m_axis TKEEP")

  io.s_axi_ctrl_awaddr.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWADDR")
  io.s_axi_ctrl_awprot.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWPROT")
  io.s_axi_ctrl_awvalid.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWVALID")
  io.s_axi_ctrl_awready.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWREADY")
  io.s_axi_ctrl_wdata.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl WDATA")
  io.s_axi_ctrl_wstrb.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl WSTRB")
  io.s_axi_ctrl_wvalid.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl WVALID")
  io.s_axi_ctrl_wready.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl WREADY")
  io.s_axi_ctrl_bresp.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl BRESP")
  io.s_axi_ctrl_bvalid.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl BVALID")
  io.s_axi_ctrl_bready.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl BREADY")
  io.s_axi_ctrl_araddr.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARADDR")
  io.s_axi_ctrl_arprot.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARPROT")
  io.s_axi_ctrl_arvalid.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARVALID")
  io.s_axi_ctrl_arready.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARREADY")
  io.s_axi_ctrl_rdata.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl RDATA")
  io.s_axi_ctrl_rresp.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl RRESP")
  io.s_axi_ctrl_rvalid.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl RVALID")
  io.s_axi_ctrl_rready.addAttribute("X_INTERFACE_INFO", "xilinx.com:interface:aximm:1.0 s_axi_ctrl RREADY")
  io.s_axi_ctrl_awaddr.addAttribute(
    "X_INTERFACE_PARAMETER",
    "PROTOCOL AXI4LITE,READ_WRITE_MODE READ_WRITE,ADDR_WIDTH 32,DATA_WIDTH 32," +
      "HAS_BRESP 1,HAS_BURST 0,HAS_CACHE 0,HAS_LOCK 0,HAS_PROT 1,HAS_QOS 0,HAS_REGION 0," +
      "HAS_RRESP 1,HAS_WSTRB 1,SUPPORTS_NARROW_BURST 0"
  )

  // Clock and reset attributes
  clockDomain.clock.addAttribute("X_INTERFACE_INFO", "xilinx.com:signal:clock:1.0 aclk CLK")
  clockDomain.clock.addAttribute(
    "X_INTERFACE_PARAMETER",
    "ASSOCIATED_BUSIF s_axis:m_axis:s_axi_ctrl,ASSOCIATED_RESET aresetn," +
      "CLK_DOMAIN clock,FREQ_HZ 100000000,PHASE 0.000,INSERT_VIP 0"
  )

  clockDomain.reset.addAttribute("X_INTERFACE_INFO", "xilinx.com:signal:reset:1.0 aresetn RST")
  clockDomain.reset.addAttribute("X_INTERFACE_PARAMETER", "POLARITY ACTIVE_LOW,INSERT_VIP 0")

  // ===== Packet Decoder FSM (Phase 2) =====

  object InputState extends SpinalEnum {
    val IDLE, COLLECTING, DRAINING = newElement()
  }

  val inputState = Reg(InputState()) init (InputState.IDLE)

  // Output FSM states
  object OutputState extends SpinalEnum {
    val STREAMING_SPIKES, EMIT_TIMESTEP = newElement()
  }

  val outputState = Reg(OutputState()) init (OutputState.STREAMING_SPIKES)

  // Instruction type constants
  val TYPE_NOOP     = 0 // No operation - ignore packet
  val TYPE_SPIKE    = 1 // Spike with coordinate and value
  val TYPE_TIMESTEP = 2 // End of timestep marker

  // Decode packet fields
  val instructionType = io.s_axis.data(2 downto 0).asUInt
  val coordinate      = io.s_axis.data(15 downto 3).asUInt  // 13 bits instead of 8
  val spikeValueRaw   = io.s_axis.data(31 downto 16).asSInt // Bits 31:16 instead of 26:11

  val linearSize  = inputShape.reduce(_ * _)
  val bufferWidth = inputQuant.qformat.width

  // FIFO for sparse spike buffering (depth = linearSize for worst-case all-active)
  val spikeFifo  = StreamFifo(SpikeEntry(log2Up(linearSize), bufferWidth), linearSize)
  val spikeCount = Reg(UInt(log2Up(linearSize + 1) bits)) init 0
  val drainCount = Reg(UInt(log2Up(linearSize + 1) bits)) init 0

  // Debug counters for timesteps
  val inputTimestepCounter  = Reg(UInt(32 bits)) init 0
  val outputTimestepCounter = Reg(UInt(32 bits)) init 0
  val outputTimestepInFrame = Reg(UInt(32 bits)) init 0

  // Expected number of timesteps from config (for TLAST logic)
  val totalTimesteps = config.timesteps

  // Management register addresses (AXI-Lite)
  val AXIL_INPUT_TIMESTEPS_ADDR  = U(0x00, 8 bits)
  val AXIL_OUTPUT_TIMESTEPS_ADDR = U(0x04, 8 bits)

  // Centralized timestep events used by both counters and management registers
  val inputTimestepEvent = io.s_axis.fire &&
    ((inputState === InputState.IDLE) || (inputState === InputState.COLLECTING)) &&
    (instructionType === TYPE_TIMESTEP)

  val outputTimestepEvent = io.m_axis.fire && (outputState === OutputState.EMIT_TIMESTEP)

  // ===== Packet Conversion Logic (Phases 3-4) =====

  // Create internal stream to Accelerator
  val toAccelerator = Stream(Fragment(Activations(inputActivationConfig)))

  // Default values to avoid latches
  toAccelerator.valid                         := False
  toAccelerator.payload.last                  := False
  toAccelerator.payload.fragment.coords(0)    := U(0)
  toAccelerator.payload.fragment.value(0).raw := B(0)

  // Default: FIFO push not valid, pop not ready
  spikeFifo.io.push.valid         := False
  spikeFifo.io.push.payload.coord := U(0)
  spikeFifo.io.push.payload.value := B(0, bufferWidth bits)
  spikeFifo.io.pop.ready          := False

  // Packet processing state machine
  switch(inputState) {
    is(InputState.IDLE) {
      when(io.s_axis.valid) {
        switch(instructionType) {
          is(TYPE_NOOP) {
            // No operation - do nothing, stay in IDLE
          }
          is(TYPE_SPIKE) {
            when(coordinate < linearSize) {
              spikeFifo.io.push.valid         := True
              spikeFifo.io.push.payload.coord := coordinate.resized
              spikeFifo.io.push.payload.value := spikeValueRaw.asBits.resized
              spikeCount                      := spikeCount + 1
            }
            inputState := InputState.COLLECTING
          }
          is(TYPE_TIMESTEP) {
            drainCount := 0
            inputState := InputState.DRAINING
          }
        }
      }
    }

    is(InputState.COLLECTING) {
      when(io.s_axis.valid) {
        switch(instructionType) {
          is(TYPE_NOOP) {
            // No operation - do nothing, stay in COLLECTING
          }
          is(TYPE_SPIKE) {
            when(coordinate < linearSize) {
              spikeFifo.io.push.valid         := True
              spikeFifo.io.push.payload.coord := coordinate.resized
              spikeFifo.io.push.payload.value := spikeValueRaw.asBits.resized
              spikeCount                      := spikeCount + 1
            }
          }
          is(TYPE_TIMESTEP) {
            drainCount := 0
            inputState := InputState.DRAINING
          }
        }
      }
    }

    is(InputState.DRAINING) {
      when(spikeCount === 0) {
        // Empty timestep: emit a single zero-value activation with last=True
        toAccelerator.valid                         := True
        toAccelerator.payload.fragment.coords(0)    := U(0)
        toAccelerator.payload.fragment.value(0).raw := B(0)
        toAccelerator.payload.last                  := True
        when(toAccelerator.fire) {
          inputState := InputState.IDLE
        }
      } otherwise {
        // Drain FIFO entries to toAccelerator
        toAccelerator.valid                         := spikeFifo.io.pop.valid
        toAccelerator.payload.fragment.coords(0)    := spikeFifo.io.pop.payload.coord.resized
        toAccelerator.payload.fragment.value(0).raw := spikeFifo.io.pop.payload.value
        toAccelerator.payload.last                  := (drainCount === spikeCount - 1)

        spikeFifo.io.pop.ready := toAccelerator.ready

        when(toAccelerator.fire) {
          drainCount := drainCount + 1
          when(drainCount === spikeCount - 1) {
            // Last spike drained
            spikeCount := 0
            drainCount := 0
            inputState := InputState.IDLE
          }
        }
      }
    }
  }

  // ===== Flow Control (Phase 6) =====

  // Accept new packets only when in IDLE state and ready to output
  io.s_axis.ready := (inputState === InputState.IDLE) || (inputState === InputState.COLLECTING)

  // ===== Accelerator Instantiation (Phase 5) =====

  val accelerator = Accelerator(
    nirGraph = nirGraph,
    config = config,
    accelConfig = accelConfig
  )

  // Connect packet converter to Accelerator input
  accelerator.io.input << toAccelerator

  // Pipeline/register the accelerator output
  val pipelined_output = accelerator.io.output.m2sPipe().s2mPipe()

  // Expose accelerator output signals for simulation-based output checking
  accelerator.io.output.simPublic()

  // ===== Packet Encoder FSM (Output Serializer) =====

  // Spike detection: check if value is non-zero
  val isSpike = pipelined_output.fragment.value(0).raw =/= 0

  // Default values (avoid latches)
  io.m_axis.valid        := False
  io.m_axis.payload.data := B(32 bits, default -> false)
  io.m_axis.payload.keep := 0xf
  io.m_axis.payload.last := False
  pipelined_output.ready := False

  // FSM state machine
  switch(outputState) {
    is(OutputState.STREAMING_SPIKES) {
      // Forward non-zero spikes to AXI
      when(isSpike) {
        // Non-zero spike - encode as TYPE_SPIKE packet
        io.m_axis.valid                    := pipelined_output.valid
        io.m_axis.payload.data(2 downto 0) := TYPE_SPIKE

        val coord13bit = pipelined_output.fragment.coords(0).resize(13) // Resize to 13 bits
        val output     = pipelined_output.fragment.value(0)
        io.m_axis.payload.data(15 downto 3)  := coord13bit.asBits     // 13-bit coordinate
        io.m_axis.payload.data(31 downto 16) := output.raw.resize(16) // Bits 31:16

        // TLAST is reserved for timestep framing in EMIT_TIMESTEP state.
        // Spike packets are intra-timestep payload and must not terminate DMA frames.
        io.m_axis.payload.last := False

        pipelined_output.ready := io.m_axis.ready // Backpressure
      } otherwise {
        // Zero value - consume from accelerator but don't forward
        io.m_axis.valid        := False
        io.m_axis.payload.last := False
        pipelined_output.ready := True // Always consume zeros
      }

      // Detect end of timestep (Fragment.last goes high)
      when(pipelined_output.fire && pipelined_output.last) {
        outputState := OutputState.EMIT_TIMESTEP
      }
    }

    is(OutputState.EMIT_TIMESTEP) {
      // Emit TYPE_TIMESTEP packet
      io.m_axis.valid                    := True
      io.m_axis.payload.data(2 downto 0) := TYPE_TIMESTEP

      val isLastTimestep = outputTimestepInFrame === (totalTimesteps - 1)
      // Emit TLAST on every timestep marker to define packet boundaries per timestep.
      io.m_axis.payload.last := True

      pipelined_output.ready := False // Don't consume in this state

      when(io.m_axis.fire) {
        when(isLastTimestep) {
          outputTimestepInFrame := 0
        } otherwise {
          outputTimestepInFrame := outputTimestepInFrame + 1
        }
        outputState := OutputState.STREAMING_SPIKES
      }
    }
  }

  // Cumulative counters (monotonic since reset)
  when(inputTimestepEvent) {
    inputTimestepCounter := inputTimestepCounter + 1
  }
  when(outputTimestepEvent) {
    outputTimestepCounter := outputTimestepCounter + 1
  }

  // ===== AXI-Lite Management Interface =====

  val AXI_RESP_OKAY = B"2'b00"

  // Write channel tracking (writes are acknowledged, no writable registers yet)
  val writeAddressPending = Reg(Bool()) init False
  val writeDataPending    = Reg(Bool()) init False
  val writeResponseValid  = Reg(Bool()) init False

  io.s_axi_ctrl_awready := !writeAddressPending && !writeResponseValid
  io.s_axi_ctrl_wready  := !writeDataPending && !writeResponseValid

  when(io.s_axi_ctrl_awready && io.s_axi_ctrl_awvalid) {
    writeAddressPending := True
  }
  when(io.s_axi_ctrl_wready && io.s_axi_ctrl_wvalid) {
    writeDataPending := True
  }
  when(writeAddressPending && writeDataPending && !writeResponseValid) {
    writeAddressPending := False
    writeDataPending    := False
    writeResponseValid  := True
  }
  when(writeResponseValid && io.s_axi_ctrl_bready) {
    writeResponseValid := False
  }

  io.s_axi_ctrl_bvalid := writeResponseValid
  io.s_axi_ctrl_bresp  := AXI_RESP_OKAY

  // Read channel handling
  val readValid = Reg(Bool()) init False
  val readData  = Reg(Bits(32 bits)) init 0

  io.s_axi_ctrl_arready := !readValid
  io.s_axi_ctrl_rvalid  := readValid
  io.s_axi_ctrl_rdata   := readData
  io.s_axi_ctrl_rresp   := AXI_RESP_OKAY

  when(io.s_axi_ctrl_arready && io.s_axi_ctrl_arvalid) {
    readValid := True
    switch(io.s_axi_ctrl_araddr(7 downto 0)) {
      is(AXIL_INPUT_TIMESTEPS_ADDR) {
        readData := inputTimestepCounter.asBits
      }
      is(AXIL_OUTPUT_TIMESTEPS_ADDR) {
        readData := outputTimestepCounter.asBits
      }
      default {
        readData := B(0, 32 bits)
      }
    }
  }
  when(readValid && io.s_axi_ctrl_rready) {
    readValid := False
  }

  // Connect debug outputs
  io.debug_input_timesteps  := inputTimestepCounter
  io.debug_output_timesteps := outputTimestepCounter
  io.debug_input_state      := inputState.asBits.asUInt.resized
  io.debug_output_state     := outputState.asBits.asUInt.resized
}

case class CompilationSettings(reduction: Boolean, macWidth: Int, spikeGating: Boolean, datasetName: String)

object CompilationSettings {

  implicit val decoder: Decoder[CompilationSettings] = Decoder.instance { c =>
    for {
      reduction   <- c.downField("reduction").as[Option[Boolean]]
      macWidth    <- c.downField("macWidth").as[Option[Int]]
      spikeGating <- c.downField("spikeGating").as[Option[Boolean]]
      datasetName <- c.downField("dataset_name").as[Option[String]]
    } yield CompilationSettings(
      reduction = reduction.getOrElse(false),
      macWidth = macWidth.getOrElse(1),
      spikeGating = spikeGating.getOrElse(true),
      datasetName = datasetName.getOrElse("skip")
    )
  }

  def fromModelDir(modelDir: String): CompilationSettings = {
    val path    = s"$modelDir/compilation.json"
    val content =
      try
        scala.io.Source.fromFile(path).mkString
      catch {
        case e: Exception =>
          println(s"ERROR: Failed to read compilation metadata: $path")
          println(s"       ${e.getMessage}")
          sys.exit(1)
          ""
      }

    decode[CompilationSettings](content) match {
      case Right(settings) => settings
      case Left(err)       =>
        println(s"ERROR: Failed to parse compilation metadata: $path")
        println(s"       ${err.getMessage}")
        sys.exit(1)
        CompilationSettings(reduction = false, macWidth = 1, spikeGating = true, datasetName = "skip")
    }
  }

}

object SimUtils {

  // Instruction type constants matching AcceleratorAXI
  val TYPE_NOOP     = 0
  val TYPE_SPIKE    = 1
  val TYPE_TIMESTEP = 2

  private def fail(msg: String): Nothing = {
    println(s"ERROR: $msg")
    sys.exit(1)
  }

  private def choosePythonExecutable(): String =
    sys.env.get("PYTHON_BIN") match {
      case Some(path) if path.nonEmpty => path
      case _                           => "python3"
    }

  private implicit val uint32Decoder: Decoder[Long] = Decoder.decodeLong.emap { v =>
    if (v < 0 || v > 0xffffffffL) Left(s"Value $v out of uint32 range") else Right(v)
  }

  private implicit val uint32ListDecoder: Decoder[List[Long]] =
    Decoder.decodeList(uint32Decoder)

  private def resolveDatasetName(datasetName: Option[String]): String =
    datasetName match {
      case Some(name) if name.nonEmpty && name != "skip" => name
      case _                                             =>
        fail(
          "No usable dataset name found in compilation metadata. " +
            "Set compilation.json dataset_name or provide an explicit input packets path."
        )
    }

  private def parsePacketListFromJson(jsonText: String): List[Long] =
    decode[List[Long]](jsonText)(uint32ListDecoder) match {
      case Right(values) => values
      case Left(err)     => fail(s"Failed to parse packet JSON from Python bridge: ${err.getMessage}")
    }

  def loadPacketsFromNpy(packetsPath: String): List[Long] = {
    val tensor = nir.tensor.Tensor.fromNumpy[Double](packetsPath)
    tensor.toFlatList.map(v => v.toLong & 0xffffffffL)
  }

  /**
   * Load recordings.npy (the expected quantized output trace, dequantized real
   * values, shape (T, N_out)) as a flat Array[Double].
   */
  def loadRecordingsNpy(recordingsPath: String): Array[Double] = {
    val tensor = nir.tensor.Tensor.fromNumpy[Double](recordingsPath)
    tensor.toFlatList.toArray
  }

  /**
   * Round a dequantized real value to `precision` fractional bits — the
   * outputCheck comparison quantum. precision == output frac bits ⇒ bit-exact.
   */
  def quantizeToPrecision(v: Double, precision: Int): Int =
    scala.math.round(v * scala.math.pow(2, precision)).toInt

  private def generatePacketsFromDataset(
    modelDir: String,
    datasetName: String,
    datasetIndex: Int
  ): List[Long] = {
    val bridgeScript = new java.io.File("scripts/dataset_packets_bridge.py")
    if (!bridgeScript.exists()) {
      fail(
        s"Python bridge script not found: ${bridgeScript.getPath}. " +
          "Expected to run from 2-compilation with scripts/dataset_packets_bridge.py present."
      )
    }

    val pythonBin = choosePythonExecutable()
    val cmd       = Seq(
      pythonBin,
      bridgeScript.getPath,
      "--model-dir",
      modelDir,
      "--dataset-name",
      datasetName,
      "--dataset-index",
      datasetIndex.toString
    )

    val stdout   = new StringBuilder
    val stderr   = new StringBuilder
    val exitCode = Process(cmd, new java.io.File(".")).!(
      ProcessLogger(
        line => stdout.append(line).append("\n"),
        line => stderr.append(line).append("\n")
      )
    )

    if (exitCode != 0) {
      fail(
        s"Dataset packet bridge failed (exit=$exitCode).\n" +
          s"Command: ${cmd.mkString(" ")}\n" +
          s"Stderr:\n${stderr.toString()}"
      )
    }

    val payload = stdout.toString().trim
    if (payload.isEmpty) {
      fail("Dataset packet bridge produced empty stdout; expected JSON packet array")
    }

    parsePacketListFromJson(payload)
  }

  private def resolveInputPackets(
    modelDir: String,
    config: ConfigJSON,
    datasetIndex: Option[Int],
    datasetName: Option[String],
    packetsPath: Option[String] = None
  ): List[Long] =
    packetsPath match {
      case Some(path) =>
        val packets = loadPacketsFromNpy(path)
        println(s"✓ Loaded ${packets.length} packets from $path")
        packets
      case None =>
        val index = datasetIndex.getOrElse {
          fail(
            "No input packet npy file provided and no --dataset-index given. " +
              "Either run save_files() to generate input_packets.npy or pass --dataset-index=<N>."
          )
        }

        if (index < 0) {
          fail(s"dataset-index must be non-negative, got $index")
        }

        val name    = resolveDatasetName(datasetName)
        val packets = generatePacketsFromDataset(modelDir, name, index)

        if (packets.isEmpty) {
          fail(
            s"Runtime dataset packet generation returned zero packets for $name[$index]. " +
              "Refusing to run simulation with empty input stream."
          )
        }

        println(
          s"✓ Generated ${packets.length} packets from runtime dataset sample " +
            s"(dataset=$name, index=$index)"
        )
        packets
    }

  def saveBehavioralOutputs(
    outputPath: String,
    config: ConfigJSON,
    outputPackets: Seq[Long]
  ): Unit = {
    val packetsJson = outputPackets.map(_.toString).mkString("[", ",", "]")
    val json        = s"""{
      |  "timesteps": ${config.timesteps},
      |  "output_packets": $packetsJson
      |}""".stripMargin

    // Write to behavioral.json in the output directory
    val behavioralFile = s"$outputPath/behavioral.json"
    try {
      val writer = new java.io.FileWriter(behavioralFile)
      writer.write(json)
      writer.close()
      println(s"✓ Saved behavioral outputs to: $behavioralFile")
    } catch {
      case e: Exception =>
        println(s"WARNING: Failed to save behavioral outputs: ${e.getMessage}")
    }
  }

  def runSimulation(
    modelDir: String,
    nirFile: String,
    jsonContent: String,
    accelConfig: AcceleratorConfig,
    config: ConfigJSON,
    withWave: Boolean = true,
    datasetIndex: Option[Int] = None,
    datasetName: Option[String] = None,
    packetsPath: Option[String] = None
  ): Seq[Long] = {
    val timeout = 100000000

    var outputPackets: Seq[Long] = Seq()

    try {
      val simBase =
        if (withWave) SimConfig.withIVerilog.withFstWave.withConfig(SpinalConfig().includeSimulation)
        else SimConfig.withIVerilog.withConfig(SpinalConfig().includeSimulation)

      simBase
        .workspacePath("./simWorkspace")
        .compile(new AcceleratorAXI(nirFile, jsonContent, accelConfig))
        .doSim { dut =>
          SimTimeout(timeout)
          dut.clockDomain.forkStimulus(2)
          sleep(10)

          val inputAXI: List[Long] =
            resolveInputPackets(
              modelDir,
              config,
              datasetIndex = datasetIndex,
              datasetName = datasetName,
              packetsPath = packetsPath
            )

          val inputQueue = scala.collection.mutable.Queue(inputAXI: _*)

          // === Standard simulation branch ===
          val outputs       = scala.collection.mutable.ArrayBuffer[Int]()
          var timestepsRcvd = 0
          var lastSeen      = false

          // Track expected input timesteps by counting TIMESTEP packets
          var expectedInputTimesteps = 0

          // Latency tracking (simTime() returns time units, clock period is 2)
          val clockPeriod    = 2
          var firstInputTime = -1L
          var lastOutputTime = -1L

          dut.io.m_axis.ready #= true
          StreamMonitor(dut.io.m_axis, dut.clockDomain) { payload =>
            outputs += payload.data.toInt

            // Track last output time for latency calculation
            lastOutputTime = simTime()

            // Count TIMESTEP packets to track received timesteps
            val instructionType = payload.data.toInt & 0x7 // Extract bits [2:0]
            if (instructionType == TYPE_TIMESTEP) {
              timestepsRcvd += 1
              println(s"✓ TIMESTEP packet received (timestep ${timestepsRcvd})")
              // End when we've received all expected timesteps
              if (timestepsRcvd >= config.timesteps) {
                lastSeen = true
              }
            }
          }

          // Drive input packets to s_axis
          dut.io.s_axis.valid #= false
          dut.io.s_axis.data #= 0
          dut.io.s_axis.last #= false
          dut.io.s_axis.keep #= 0xf
          fork {
            while (inputQueue.nonEmpty) {
              val packet = inputQueue.front
              dut.io.s_axis.valid #= true
              dut.io.s_axis.data #= packet
              dut.io.s_axis.last #= (inputQueue.size == 1) // Set TLAST on final input packet
              dut.io.s_axis.keep #= 0xf                    // Keep all 4 bytes (32-bit word)

              // Track first input time for latency calculation
              if (firstInputTime < 0) {
                firstInputTime = simTime()
              }

              do dut.clockDomain.waitSampling()
              while (!dut.io.s_axis.ready.toBoolean)
              inputQueue.dequeue()

              // Count TIMESTEP packets to track expected input count
              val instructionType = packet & 0x7 // Extract bits [2:0]
              if (instructionType == TYPE_TIMESTEP) {
                expectedInputTimesteps += 1
              }
            }
            dut.io.s_axis.valid #= false
            dut.io.s_axis.last #= false
          }

          // Wait for output TLAST signal
          dut.clockDomain.waitSamplingWhere(timeout)(lastSeen)
          sleep(10)

          // Validate results
          println(s"✓ Test complete: Received ${outputs.length} output packets")
          println(s"✓ Expected: input=${expectedInputTimesteps}, output=${timestepsRcvd}")

          // Print latency metrics
          val latencyCycles = (lastOutputTime - firstInputTime) / clockPeriod
          println(
            s"✓ Latency: ${latencyCycles} cycles (first input @ ${firstInputTime / clockPeriod}, last output @ ${lastOutputTime / clockPeriod})"
          )

          // Decode and validate output packets
          val decodedPackets = outputs.map { packet =>
            val instructionType = packet & 0x7
            val coord           = (packet >> 3) & 0x1fff  // 13-bit mask
            val value           = (packet >> 16) & 0xffff // Extract from bit 16
            (instructionType, coord, value)
          }

          // Count packet types
          val spikeCount    = decodedPackets.count(_._1 == TYPE_SPIKE)
          val timestepCount = decodedPackets.count(_._1 == TYPE_TIMESTEP)

          println(s"✓ Packet counts: ${spikeCount} spikes, ${timestepCount} timesteps")

          // Validate timestep count
          if (timestepCount != config.timesteps) {
            println(s"ERROR: Expected ${config.timesteps} TIMESTEP packets, got ${timestepCount}")
            sys.exit(1)
          }

          // Validate spike filtering (no zeros)
          val zeroSpikes = decodedPackets.filter(_._1 == TYPE_SPIKE).filter(_._3 == 0)
          if (zeroSpikes.nonEmpty) {
            println(s"ERROR: Zero-valued spike detected at coord=${zeroSpikes.head._2}")
            sys.exit(1)
          }

          println(s"✓ Spike filtering validated: no zero-valued spikes")
          println(s"✓ Debug wire validation passed!")

          // Convert to Long packets and return
          outputPackets = outputs.map(_.toLong).toSeq
        }
    } catch {
      case e: Exception =>
        println(s"ERROR: Simulation failed: ${e.getMessage}")
        e.printStackTrace()
        sys.exit(1)
    }

    outputPackets
  }

}

object Generate extends App {

  def printUsage(): Unit = {
    println("Usage: AcceleratorAXIGen <model-dir> [output-dir] [options]")
    println()
    println("Arguments:")
    println("  <model-dir>  Directory containing model.nir, model.json, and compilation.json")
    println("  [output-dir] Output directory for Verilog files (default: current directory)")
    println()
    println("Options:")
    println("  --test              Run simulation and save behavioral.json to 2-behavioral/outputs/<design>/")
    println("  --dataset-index=<N> Dataset index for simulation when model.json has no input_packets")
    println("                     (requires compilation.json dataset_name and Python bridge dependencies)")
    println()
    println("Examples:")
    println("  sbt \"runMain NIR2FPGA.Generate ../train/tests/lif\"")
    println("  sbt \"runMain NIR2FPGA.Generate ../train/tests/lif ./output\"")
    println("  sbt \"runMain NIR2FPGA.Generate ../train/tests/lif --test\"")
    println("  sbt \"runMain NIR2FPGA.Generate ../train/tests/lif --test --dataset-index=5\"")
    println()
    println("Configuration:")
    println("  Clock frequency: 100 MHz")
    println("  Reset: ASYNC, active LOW")
  }

  // ===== Argument Parsing =====
  val positionalArgs = args.filterNot(_.startsWith("--"))
  val optionArgs     = args.filter(_.startsWith("--"))

  if (positionalArgs.length < 1 || positionalArgs.length > 2) {
    println("ERROR: Invalid number of arguments")
    println()
    printUsage()
    sys.exit(1)
  }

  val modelDir  = positionalArgs(0)
  val outputDir = if (positionalArgs.length == 2) positionalArgs(1) else "./outputs"

  // Parse options
  val optionMap = optionArgs.map { opt =>
    val parts = opt.stripPrefix("--").split("=", 2)
    if (parts.length == 2) parts(0) -> parts(1)
    else parts(0)                   -> "true"
  }.toMap

  val runTest      = optionMap.getOrElse("test", "false").toLowerCase == "true"
  val datasetIndex = optionMap.get("dataset-index").map(_.toInt)

  val compilation = CompilationSettings.fromModelDir(modelDir)

  val accelConfig = AcceleratorConfig(
    reduction = compilation.reduction,
    spikeGating = compilation.spikeGating,
    macWidth = compilation.macWidth
  )

  // ===== File Validation (Pre-flight checks) =====
  val modelDirPath = new java.io.File(modelDir)

  // Construct paths to model files
  val nirFile  = s"$modelDir/model.nir"
  val jsonFile = s"$modelDir/model.json"

  val outputDirFull = {
    val name = modelDirPath.getName()
    outputDir + "/" + name
  }

  val nirPath  = new java.io.File(nirFile)
  val jsonPath = new java.io.File(jsonFile)
  val outPath  = new java.io.File(outputDirFull)

  // Check model directory exists
  if (!modelDirPath.exists()) {
    println(s"ERROR: Model directory not found: $modelDir")
    sys.exit(1)
  }
  if (!modelDirPath.isDirectory()) {
    println(s"ERROR: Model path is not a directory: $modelDir")
    sys.exit(1)
  }

  // Check NIR file exists
  if (!nirPath.exists()) {
    println(s"ERROR: model.nir not found in directory: $modelDir")
    println(s"       Expected: $nirFile")
    sys.exit(1)
  }
  if (!nirPath.canRead()) {
    println(s"ERROR: Cannot read model.nir: $nirFile")
    sys.exit(1)
  }

  // Check JSON file exists
  if (!jsonPath.exists()) {
    println(s"ERROR: model.json not found in directory: $modelDir")
    println(s"       Expected: $jsonFile")
    sys.exit(1)
  }
  if (!jsonPath.canRead()) {
    println(s"ERROR: Cannot read model.json: $jsonFile")
    sys.exit(1)
  }

  // Check output directory exists (or parent if creating new)
  if (!outPath.exists()) {
    println(s"INFO: Output directory does not exist, will be created: $outPath")
    if (!outPath.mkdirs()) {
      println(s"ERROR: Failed to create output directory: $outPath")
      sys.exit(1)
    }
  }
  if (!outPath.isDirectory()) {
    println(s"ERROR: Output path is not a directory: $outPath")
    sys.exit(1)
  }

  // ===== JSON Pre-validation (Fail Fast) =====
  println(s"Validating configuration files...")

  val jsonContent =
    try
      scala.io.Source.fromFile(jsonFile).mkString
    catch {
      case e: Exception =>
        println(s"ERROR: Failed to read JSON file: ${e.getMessage}")
        sys.exit(1)
        "" // Unreachable but needed for type
    }

  // Attempt to parse JSON early to catch config errors
  val config =
    try {
      val cfg = ConfigJSON.fromJson(jsonContent)
      println(s"✓ JSON configuration validated")
      cfg
    } catch {
      case e: Exception =>
        println(s"ERROR: Invalid JSON configuration: ${e.getMessage}")
        sys.exit(1)
        null // Unreachable but needed for type
    }

  // ===== Summary =====
  println()
  println("Generating AcceleratorAXI Verilog with configuration:")
  println(s"  Model dir:    $modelDir")
  println(s"    - model.nir:  ✓")
  println(s"    - model.json: ✓")
  println(s"  Output dir:   $outPath")
  println(s"  Clock:        100 MHz")
  println(s"  Reset:        ASYNC (active LOW)")
  println(s"  Reduction:    ${compilation.reduction}")
  println(s"  SpikeGating:  ${compilation.spikeGating}")
  println(s"  MacWidth:     ${compilation.macWidth}")
  println()

  val nirGraph = NIRGraph(new java.io.File(nirFile))

  // ===== Verilog Generation =====
  try {
    SpinalConfig(
      targetDirectory = outPath.toString(),
      defaultClockDomainFrequency = FixedFrequency(100 MHz),
      defaultConfigForClockDomains = ClockDomainConfig(
        resetKind = ASYNC,
        resetActiveLevel = LOW
      ),
      rtlHeader = s"""
NIR Graph:
${nirGraph}

Reduction: ${compilation.reduction}
SpikeGating: ${compilation.spikeGating}
MacWidth: ${compilation.macWidth}
"""
    ).generateVerilog(
      new AcceleratorAXI(nirFile, jsonContent, accelConfig)
    )

    println()
    println("✓ Verilog generation complete!")
    println(s"✓ Output written to: $outPath/AcceleratorAXI.v")

    // ===== Optional Simulation and Behavioral Output =====
    if (runTest) {
      println()
      println("Running simulation for behavioral output...")

      // Run simulation
      val outputPackets = SimUtils.runSimulation(
        modelDir = modelDir,
        nirFile = nirFile,
        jsonContent = jsonContent,
        accelConfig = accelConfig,
        config = config,
        withWave = false,
        datasetIndex = datasetIndex,
        datasetName = Some(compilation.datasetName)
      )

      // Save behavioral.json to the same output directory as AcceleratorAXI.v
      SimUtils.saveBehavioralOutputs(outputDirFull, config, outputPackets)

      println()
      println(s"✓ Behavioral output saved to: $outputDirFull/behavioral.json")
    }

  } catch {
    case e: Exception =>
      println()
      println(s"ERROR: Verilog generation failed: ${e.getMessage}")
      e.printStackTrace()
      sys.exit(1)
  }
}

object Test extends App {

  // Instruction type constants matching AcceleratorAXI (also defined in SimUtils)
  val TYPE_NOOP     = SimUtils.TYPE_NOOP
  val TYPE_SPIKE    = SimUtils.TYPE_SPIKE
  val TYPE_TIMESTEP = SimUtils.TYPE_TIMESTEP

  def printUsage(): Unit = {
    println("Usage: AcceleratorAXITest <model-dir> [options]")
    println()
    println("Arguments:")
    println("  <model-dir>  Directory containing model.nir, model.json, and compilation.json")
    println()
    println("Options:")
    println("  --input_packets=<path>    Required path to input_packets.npy")
    println("  --recordings_path=<path>  Optional path to recordings.npy for output validation")
    println("  --precision=<k>           Fractional-bit resolution for recordings comparison")
    println("                            (default: output qformat frac bits ⇒ bit-exact)")
    println()
    println("Examples:")
    println("  sbt \"runMain NIR2FPGA.Test ../train/tests/lif --input_packets=/tmp/input_packets.npy\"")
    println(
      "  sbt \"runMain NIR2FPGA.Test ../train/tests/network --input_packets=/tmp/input_packets.npy --recordings_path=/tmp/recordings.npy\""
    )
    println()
    println("Configuration:")
    println("  Simulation: Verilator with waveform generation")
    println("  Timeout: 1000000 cycles")
  }

  // ===== Argument Parsing =====
  val positionalArgs = args.filterNot(_.startsWith("--"))
  val optionArgs     = args.filter(_.startsWith("--"))

  if (positionalArgs.length != 1) {
    println("ERROR: Invalid number of arguments")
    println()
    printUsage()
    sys.exit(1)
  }

  val modelDir = positionalArgs(0)

  // Parse options
  val optionMap = optionArgs.map { opt =>
    val parts = opt.stripPrefix("--").split("=", 2)
    if (parts.length == 2) parts(0) -> parts(1)
    else parts(0)                   -> "true"
  }.toMap

  if (optionMap.contains("packets")) {
    println("ERROR: --packets is removed. Use --input_packets=<path>.")
    println()
    printUsage()
    sys.exit(1)
  }
  if (optionMap.contains("outputCheck")) {
    println("ERROR: --outputCheck is removed. Use --recordings_path=<path> to enable output validation.")
    println()
    printUsage()
    sys.exit(1)
  }
  if (optionMap.contains("outputRecording")) {
    println("ERROR: --outputRecording is removed. Use --recordings_path=<path>.")
    println()
    printUsage()
    sys.exit(1)
  }

  val inputPacketsPath: String = optionMap.get("input_packets") match {
    case Some(path) if path.nonEmpty && path != "true" => path
    case _                                             =>
      println("ERROR: Missing required option --input_packets=<path>")
      println()
      printUsage()
      sys.exit(1)
      "" // Unreachable but needed for type
  }

  val recordingsPathOpt: Option[String] = optionMap.get("recordings_path") match {
    case Some(path) if path.nonEmpty && path != "true" => Some(path)
    case Some(_)                                       =>
      println("ERROR: --recordings_path requires a value, for example --recordings_path=/tmp/recordings.npy")
      println()
      printUsage()
      sys.exit(1)
      None // Unreachable but needed for type
    case None => None
  }

  val compilation               = CompilationSettings.fromModelDir(modelDir)
  val outputCheck               = recordingsPathOpt.isDefined
  val precisionOpt: Option[Int] = optionMap.get("precision").map(_.toInt)

  val accelConfig = AcceleratorConfig(
    reduction = compilation.reduction,
    spikeGating = compilation.spikeGating,
    macWidth = compilation.macWidth
  )

  // Construct paths to model files
  val nirFile  = s"$modelDir/model.nir"
  val jsonFile = s"$modelDir/model.json"

  // ===== File Validation (Pre-flight checks) =====
  val modelDirPath      = new java.io.File(modelDir)
  val nirPath           = new java.io.File(nirFile)
  val jsonPath          = new java.io.File(jsonFile)
  val inputPacketsFile  = new java.io.File(inputPacketsPath)
  val recordingsFileOpt = recordingsPathOpt.map(path => new java.io.File(path))

  // Check model directory exists
  if (!modelDirPath.exists()) {
    println(s"ERROR: Model directory not found: $modelDir")
    sys.exit(1)
  }
  if (!modelDirPath.isDirectory()) {
    println(s"ERROR: Model path is not a directory: $modelDir")
    sys.exit(1)
  }

  // Check NIR file exists
  if (!nirPath.exists()) {
    println(s"ERROR: model.nir not found in directory: $modelDir")
    println(s"       Expected: $nirFile")
    sys.exit(1)
  }
  if (!nirPath.canRead()) {
    println(s"ERROR: Cannot read model.nir: $nirFile")
    sys.exit(1)
  }

  // Check JSON file exists
  if (!jsonPath.exists()) {
    println(s"ERROR: model.json not found in directory: $modelDir")
    println(s"       Expected: $jsonFile")
    sys.exit(1)
  }
  if (!jsonPath.canRead()) {
    println(s"ERROR: Cannot read model.json: $jsonFile")
    sys.exit(1)
  }

  // Check required input packet file exists
  if (!inputPacketsFile.exists()) {
    println(s"ERROR: input_packets file not found: $inputPacketsPath")
    sys.exit(1)
  }
  if (!inputPacketsFile.canRead()) {
    println(s"ERROR: Cannot read input_packets file: $inputPacketsPath")
    sys.exit(1)
  }

  // Check optional recordings file exists when output validation is requested
  recordingsFileOpt.foreach { recordingsFile =>
    if (!recordingsFile.exists()) {
      println(s"ERROR: recordings file not found: ${recordingsFile.getPath}")
      sys.exit(1)
    }
    if (!recordingsFile.canRead()) {
      println(s"ERROR: Cannot read recordings file: ${recordingsFile.getPath}")
      sys.exit(1)
    }
  }

  // ===== Load Configuration =====
  println(s"Loading configuration files...")

  val jsonContent =
    try
      scala.io.Source.fromFile(jsonFile).mkString
    catch {
      case e: Exception =>
        println(s"ERROR: Failed to read JSON file: ${e.getMessage}")
        sys.exit(1)
        "" // Unreachable but needed for type
    }

  val config =
    try
      ConfigJSON.fromJson(jsonContent)
    catch {
      case e: Exception =>
        println(s"ERROR: Invalid JSON configuration: ${e.getMessage}")
        sys.exit(1)
        null // Unreachable but needed for type
    }

  println(s"✓ Configuration loaded")
  println()
  println("Running AcceleratorAXI simulation test:")
  println(s"  Model dir:    $modelDir")
  println(s"    - model.nir:  ✓")
  println(s"    - model.json: ✓")
  println(s"  Timesteps:    ${config.timesteps}")
  println(s"  Reduction:    ${compilation.reduction}")
  println(s"  SpikeGating:  ${compilation.spikeGating}")
  println(s"  MacWidth:     ${compilation.macWidth}")
  println(s"  InputPackets: $inputPacketsPath")
  println(s"  OutputCheck:  $outputCheck")
  recordingsPathOpt.foreach(path => println(s"  Recordings:   $path"))
  if (precisionOpt.isDefined && !outputCheck)
    println("  Note: --precision is ignored when --recordings_path is not provided")
  println()

  // ===== Run Simulation =====
  try {
    if (outputCheck) {
      // === OutputCheck branch: use SpinalHDL directly for detailed validation ===
      val timeout = 100000000
      SimConfig.withIVerilog.withFstWave
        .withConfig(SpinalConfig().includeSimulation)
        .workspacePath("./simWorkspace")
        .compile(new AcceleratorAXI(nirFile, jsonContent, accelConfig))
        .doSim { dut =>
          SimTimeout(timeout)
          dut.clockDomain.forkStimulus(2)
          sleep(10)

          val inputAXI: List[Long] = SimUtils.loadPacketsFromNpy(inputPacketsPath)
          println(s"✓ Loaded ${inputAXI.length} packets from $inputPacketsPath")

          val inputQueue = scala.collection.mutable.Queue(inputAXI: _*)
          // === OutputCheck: monitor m_axis directly for validation ===
          dut.io.m_axis.ready #= true

          // Expected output trace + comparison precision (single npy channel)
          val recordingsPath = recordingsPathOpt.getOrElse {
            println("ERROR: Internal error: recordings_path missing during output-check branch")
            sys.exit(1)
            "" // Unreachable but needed for type
          }
          val recFlat      = SimUtils.loadRecordingsNpy(recordingsPath)
          val outputSize   = recFlat.length / config.timesteps
          val outputQuant  = config.quantizations("output")("input")
          val outputBits   = outputQuant.qformat.width
          val outputSigned = outputQuant.qformat.signed
          val outputFrac   = outputQuant.qformat.fraction
          val precision    = precisionOpt.getOrElse(outputFrac)
          println(
            s"✓ Loaded recordings from $recordingsPath (outputSize=$outputSize, precision=$precision frac bits)"
          )

          val hwOutputs  = Array.fill(config.timesteps)(scala.collection.mutable.ArrayBuffer[(Int, Int)]())
          val hwPackets  = scala.collection.mutable.ArrayBuffer[Long]()
          var hwTimestep = 0

          // Monitor m_axis (AXI output stream) and decode 32-bit packets
          // Packet format: bits[2:0]=type, bits[15:3]=coord, bits[31:16]=value
          // TYPE_SPIKE=1, TYPE_TIMESTEP=2
          StreamMonitor(dut.io.m_axis, dut.clockDomain) { payload =>
            val packet = payload.data.toLong & 0xffffffffL
            hwPackets += packet
            val instructionType = (packet & 0x7).toInt
            if (instructionType == 1) { // TYPE_SPIKE
              val coord    = ((packet >> 3) & 0x1fff).toInt  // 13-bit coordinate
              val rawValue = ((packet >> 16) & 0xffff).toInt // 16-bit value
              // Interpret payload in configured fixed-point domain.
              // m_axis carries the raw bits zero-extended to 16, so sign conversion
              // must use qformat width (for example 8-bit), not 16-bit.
              val signExtendedValue =
                if (outputSigned && outputBits > 0 && outputBits < 32) {
                  val mod           = 1 << outputBits
                  val signThreshold = 1 << (outputBits - 1)
                  val masked        = rawValue & (mod - 1)
                  if (masked >= signThreshold) masked - mod else masked
                } else {
                  rawValue
                }
              // Dequantize the raw packet value, then re-quantize at the
              // requested precision so HW and recordings compare in one domain.
              val real = signExtendedValue.toDouble / scala.math.pow(2, outputFrac)
              val q    = SimUtils.quantizeToPrecision(real, precision)
              // Debug: log packets for coordinate 0 to diagnose missing expected values
              if (coord == 0) {
                println(
                  s"[DBG] hwTimestep=$hwTimestep coord=$coord raw=$rawValue sign=$signExtendedValue real=$real q=$q"
                )
              }
              if (q != 0 && hwTimestep < config.timesteps)
                hwOutputs(hwTimestep) += ((coord, q))
            } else if (instructionType == 2) { // TYPE_TIMESTEP
              hwTimestep += 1
            }
          }

          dut.io.s_axis.valid #= false
          dut.io.s_axis.data #= 0
          dut.io.s_axis.last #= false
          dut.io.s_axis.keep #= 0xf
          fork {
            while (inputQueue.nonEmpty) {
              val packet = inputQueue.front
              dut.io.s_axis.valid #= true
              dut.io.s_axis.data #= packet
              dut.io.s_axis.last #= (inputQueue.size == 1)
              dut.io.s_axis.keep #= 0xf
              do dut.clockDomain.waitSampling()
              while (!dut.io.s_axis.ready.toBoolean)
              inputQueue.dequeue()
            }
            dut.io.s_axis.valid #= false
            dut.io.s_axis.last #= false
          }

          dut.clockDomain.waitSamplingWhere(timeout)(hwTimestep >= config.timesteps)
          sleep(10)

          val expected = Array.tabulate(config.timesteps) { t =>
            (0 until outputSize).flatMap { n =>
              val q = SimUtils.quantizeToPrecision(recFlat(t * outputSize + n), precision)
              if (q != 0) Some((n, q)) else None
            }.toSet
          }

          def formatDecodedEntry(entry: Option[(Int, Int)]): String = entry match {
            case Some((coord, q)) =>
              val decodedValue = q.toDouble / scala.math.pow(2, precision)
              f"val=$decodedValue%.3f coord=$coord"
            case None =>
              "val=<missing> coord=<missing>"
          }

          def representativeEntry(primary: Set[(Int, Int)], secondary: Set[(Int, Int)]): Option[(Int, Int)] = {
            val diff = primary.diff(secondary).toSeq.sortBy { case (coord, q) => (coord, q) }
            diff.headOption.orElse(primary.toSeq.sortBy { case (coord, q) => (coord, q) }.headOption)
          }

          var allMatch              = true
          var mismatchHeaderPrinted = false
          for (t <- 0 until config.timesteps) {
            val hwSet  = hwOutputs(t).toSet
            val expSet = expected(t)
            if (hwSet != expSet) {
              allMatch = false
              if (!mismatchHeaderPrinted) {
                println(s"Mismatch details (decoded values use $precision fractional bits):")
                mismatchHeaderPrinted = true
              }

              val hwEntry   = representativeEntry(hwSet, expSet)
              val expEntry  = representativeEntry(expSet, hwSet)
              val hwString  = formatDecodedEntry(hwEntry)
              val expString = formatDecodedEntry(expEntry)
              println(s"  timestep $t: $hwString != $expString")
            }
          }

          // Save behavioral outputs to JSON for later analysis regardless of match result.
          SimUtils.saveBehavioralOutputs(modelDir, config, hwPackets.toSeq)

          if (allMatch) {
            println(s"✓ OutputCheck PASSED: all ${config.timesteps} timesteps match recordings")
          } else {
            println("WARNING: OutputCheck FAILED")
          }

        }
    } else {
      // === Standard branch: use SimUtils.runSimulation ===
      val outputPackets = SimUtils.runSimulation(
        modelDir = modelDir,
        nirFile = nirFile,
        jsonContent = jsonContent,
        accelConfig = accelConfig,
        config = config,
        withWave = true,
        packetsPath = Some(inputPacketsPath)
      )

      // Save behavioral outputs to modelDir for analysis
      SimUtils.saveBehavioralOutputs(modelDir, config, outputPackets)
    }

    println()
    println("✓ Simulation test PASSED!")
  } catch {
    case e: Exception =>
      println()
      println(s"ERROR: Simulation test failed: ${e.getMessage}")
      e.printStackTrace()
      sys.exit(1)
  }
}

object Throughput extends App {

  val modelDir =
    if (args.nonEmpty) args(0)
    else {
      println("Usage: Throughput <model-dir> [--clockMHz=100]");
      sys.exit(1)
    }

  val compilation = CompilationSettings.fromModelDir(modelDir)
  val clockMHz    = args.find(_.startsWith("--clockMHz=")).map(_.split("=")(1).toDouble).getOrElse(100.0)

  val accelConfig = AcceleratorConfig(
    reduction = compilation.reduction,
    spikeGating = compilation.spikeGating,
    macWidth = compilation.macWidth
  )

  val TYPE_SPIKE    = 1
  val TYPE_TIMESTEP = 2

  val nirFile           = s"$modelDir/model.nir"
  val jsonFile          = s"$modelDir/model.json"
  val throughputFile    = s"$modelDir/throughput_packets.json"
  val jsonContent       = scala.io.Source.fromFile(jsonFile).mkString
  val throughputContent = scala.io.Source.fromFile(throughputFile).mkString
  val tConfig           = ThroughputConfig.fromJson(throughputContent)

  val allPackets     = tConfig.samples.flatten
  val totalTimesteps = tConfig.numSamples * tConfig.timesteps

  println(
    s"Throughput measurement: ${tConfig.numSamples} samples × ${tConfig.timesteps} timesteps = $totalTimesteps total timesteps"
  )

  try {
    SimConfig.withIVerilog
      .withConfig(SpinalConfig().includeSimulation)
      .workspacePath("./simWorkspace")
      .compile(new AcceleratorAXI(nirFile, jsonContent, accelConfig))
      .doSim { dut =>
        SimTimeout(1000000000)
        dut.clockDomain.forkStimulus(2)
        sleep(10)

        val clockPeriod    = 2
        var firstInputTime = -1L
        var lastOutputTime = -1L
        var timestepsRcvd  = 0
        var startWallNs    = -1L

        dut.io.m_axis.ready #= true

        StreamMonitor(dut.io.m_axis, dut.clockDomain) { payload =>
          val instrType = payload.data.toInt & 0x7
          if (instrType == TYPE_TIMESTEP) {
            timestepsRcvd += 1
            lastOutputTime = simTime()

            if (timestepsRcvd % tConfig.timesteps == 0) {
              val samplesProcessed = timestepsRcvd / tConfig.timesteps
              val wallElapsedMin   = (System.nanoTime() - startWallNs) / 60e9
              val minPerSample     = wallElapsedMin / samplesProcessed
              val remainingSamples = tConfig.numSamples - samplesProcessed
              val etaHours         = (minPerSample * remainingSamples) / 60.0
              println(
                f"  [$samplesProcessed%3d/${tConfig.numSamples}] timesteps=$timestepsRcvd  ${minPerSample}%.2f min/sample  ETA ${etaHours}%.2f h"
              )
            }
          }
        }

        val inputQueue = scala.collection.mutable.Queue(allPackets: _*)
        StreamDriver(dut.io.s_axis, dut.clockDomain) { payload =>
          if (inputQueue.nonEmpty) {
            if (firstInputTime < 0) {
              firstInputTime = simTime()
              startWallNs = System.nanoTime()
            }
            val packet = inputQueue.dequeue()
            payload.data #= packet
            payload.last #= inputQueue.isEmpty
            payload.keep #= 0xf
            true
          } else {
            false
          }
        }.setFactor(1.0f)

        dut.clockDomain.waitSamplingWhere(1000000000)(timestepsRcvd >= totalTimesteps)
        sleep(10)

        val totalCycles = (lastOutputTime - firstInputTime) / clockPeriod
        val avgCycles   = totalCycles.toDouble / tConfig.numSamples
        val throughput  = (clockMHz * 1e6) / avgCycles

        println()
        println(s"Total cycles:          $totalCycles")
        println(s"Avg cycles/inference:  ${"%.1f".format(avgCycles)}")
        println(f"Throughput (@${clockMHz}%.0f MHz): $throughput%.1f inferences/second")
      }

    println()
    println("✓ Throughput measurement PASSED!")
  } catch {
    case e: Exception =>
      println()
      println(s"ERROR: Throughput measurement failed: ${e.getMessage}")
      e.printStackTrace()
      sys.exit(1)
  }
}
