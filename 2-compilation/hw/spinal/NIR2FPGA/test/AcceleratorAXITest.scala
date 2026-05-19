package NIR2FPGA.test

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim.{StreamDriver, StreamMonitor, StreamReadyRandomizer}
import scala.collection.mutable.{ArrayBuffer, Queue}
import scala.io.Source
import java.io.File
import nir._
import NIR2FPGA.{AcceleratorAXI, AcceleratorConfig, ConfigJSON, SimUtils}

class AcceleratorAXITest extends AnyFunSuite with Matchers {

  // Instruction type constants matching AcceleratorAXI
  val TYPE_NOOP     = 0
  val TYPE_SPIKE    = 1
  val TYPE_TIMESTEP = 2

  val AXIL_INPUT_TIMESTEPS_ADDR  = 0x00
  val AXIL_OUTPUT_TIMESTEPS_ADDR = 0x04

  private def initAxiLiteIdle(dut: AcceleratorAXI): Unit = {
    dut.io.s_axi_ctrl_awaddr #= 0
    dut.io.s_axi_ctrl_awprot #= 0
    dut.io.s_axi_ctrl_awvalid #= false
    dut.io.s_axi_ctrl_wdata #= 0
    dut.io.s_axi_ctrl_wstrb #= 0
    dut.io.s_axi_ctrl_wvalid #= false
    dut.io.s_axi_ctrl_bready #= true
    dut.io.s_axi_ctrl_araddr #= 0
    dut.io.s_axi_ctrl_arprot #= 0
    dut.io.s_axi_ctrl_arvalid #= false
    dut.io.s_axi_ctrl_rready #= true
  }

  private def axiLiteRead(dut: AcceleratorAXI, addr: Int, timeoutCycles: Int = 20000): BigInt = {
    dut.io.s_axi_ctrl_araddr #= addr
    dut.io.s_axi_ctrl_arprot #= 0
    dut.io.s_axi_ctrl_arvalid #= true

    var waited = 0
    while (waited < timeoutCycles && !dut.io.s_axi_ctrl_arready.toBoolean) {
      dut.clockDomain.waitSampling()
      waited += 1
    }
    assert(waited < timeoutCycles, s"AXI-Lite read timeout waiting for ARREADY at 0x${addr.toHexString}")

    dut.clockDomain.waitSampling()
    dut.io.s_axi_ctrl_arvalid #= false

    waited = 0
    while (waited < timeoutCycles && !dut.io.s_axi_ctrl_rvalid.toBoolean) {
      dut.clockDomain.waitSampling()
      waited += 1
    }
    assert(waited < timeoutCycles, s"AXI-Lite read timeout waiting for RVALID at 0x${addr.toHexString}")

    val data = dut.io.s_axi_ctrl_rdata.toBigInt
    dut.clockDomain.waitSampling()
    data
  }

  def runAcceleratorTest(nirFile: String, jsonFile: String): Unit = {
    val config = ConfigJSON.fromJson(jsonFile)

    val timeout = 100000000 // 50M cycles at period 2
    SimConfig.withVerilator
      .workspacePath("./simWorkspace")
      .compile(new AcceleratorAXI(nirFile, jsonFile))
      .doSim { dut =>
        SimTimeout(timeout)
        dut.clockDomain.forkStimulus(2)
        sleep(10)
        initAxiLiteIdle(dut)

        // Use pre-generated packets from Python IOManager (optional for backward compatibility)
        val inputAXI: List[Long] = config.inputPackets match {
          case Some(packets) => packets
          case None          => List()
        }

        if (inputAXI.nonEmpty) {
          println(s"✓ Loaded ${inputAXI.length} pre-generated packets from ConfigJSON")
        } else {
          println("⚠ Warning: No input packets in ConfigJSON")
        }
        // Create input queue for StreamDriver
        val inputQueue = scala.collection.mutable.Queue(inputAXI: _*)

        // Monitor outputs and track TLAST
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
        // dut.io.output_fifo_almost_full #= false // No backpressure in this test
        StreamMonitor(dut.io.m_axis, dut.clockDomain) { payload =>
          val packet          = payload.data.toInt
          val instructionType = packet & 0x7
          outputs += packet

          // Track last output time for latency calculation
          lastOutputTime = simTime()

          // Count TIMESTEP packets (not TLAST signals, since TLAST now only fires
          // on final timestep when output_fifo_almost_full is low)
          if (instructionType == TYPE_TIMESTEP) {
            timestepsRcvd += 1
            if (timestepsRcvd >= config.timesteps) {
              lastSeen = true
            }
          }

          if (payload.last.toBoolean) {
            println(s"✓ TLAST detected on m_axis (timestep ${timestepsRcvd})")
          }
        }

        // Drive input packets to s_axis
        StreamDriver(dut.io.s_axis, dut.clockDomain) { payload =>
          if (inputQueue.nonEmpty) {
            val packet = inputQueue.dequeue()
            payload.data #= packet
            payload.last #= inputQueue.isEmpty // Set TLAST on final input packet
            payload.keep #= 0xf                // Keep all 4 bytes (32-bit word)

            // Track first input time for latency calculation
            if (firstInputTime < 0) {
              firstInputTime = simTime()
            }

            // Count TIMESTEP packets to track expected input count
            val instructionType = packet & 0x7 // Extract bits [2:0]
            if (instructionType == TYPE_TIMESTEP) {
              expectedInputTimesteps += 1
            }

            true // Valid data available
          } else {
            false // No more data
          }
        }

        // Wait for output TLAST signal
        dut.clockDomain.waitSamplingWhere(timeout)(lastSeen)
        sleep(10)

        // Read debug counter values
        val debugInputTimesteps  = dut.io.debug_input_timesteps.toBigInt
        val debugOutputTimesteps = dut.io.debug_output_timesteps.toBigInt
        val axilInputTimesteps   = axiLiteRead(dut, AXIL_INPUT_TIMESTEPS_ADDR)
        val axilOutputTimesteps  = axiLiteRead(dut, AXIL_OUTPUT_TIMESTEPS_ADDR)

        // Validate results
        println(s"✓ Test complete: Received ${outputs.length} output packets")
        println(s"✓ Debug counters: input=${debugInputTimesteps}, output=${debugOutputTimesteps}")
        println(s"✓ AXI-Lite counters: input=${axilInputTimesteps}, output=${axilOutputTimesteps}")
        println(s"✓ Expected: input=${expectedInputTimesteps}, output=${timestepsRcvd}")

        // Print latency metrics
        val latencyCycles = (lastOutputTime - firstInputTime) / clockPeriod
        println(
          s"✓ Latency: ${latencyCycles} cycles (first input @ ${firstInputTime / clockPeriod}, last output @ ${lastOutputTime / clockPeriod})"
        )

        // Check debug wires match expected values
        withClue(s"Input timestep counter after ${outputs.length} packets: ") {
          debugInputTimesteps.toInt shouldBe expectedInputTimesteps
        }
        withClue(s"Output timestep counter after ${outputs.length} packets: ") {
          debugOutputTimesteps.toInt shouldBe timestepsRcvd
        }
        withClue(s"AXI-Lite input timestep register after ${outputs.length} packets: ") {
          axilInputTimesteps.toInt shouldBe expectedInputTimesteps
        }
        withClue(s"AXI-Lite output timestep register after ${outputs.length} packets: ") {
          axilOutputTimesteps.toInt shouldBe timestepsRcvd
        }

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
        withClue(s"TIMESTEP packet count (${spikeCount} spikes received): ") {
          timestepCount shouldBe config.timesteps
        }

        // Validate spike filtering (no zeros)
        decodedPackets.filter(_._1 == TYPE_SPIKE).foreach { case (typ, coord, value) =>
          withClue(s"Spike at coord=$coord should be non-zero: ") {
            value should not be 0
          }
        }

        println(s"✓ Spike filtering validated: no zero-valued spikes")
        println(s"✓ Debug wire validation passed!")
      }
  }

  test("IF only") {
    val nirFile    = "../1-internal-simulation/outputs/if/model.nir"
    val jsonString = "../1-internal-simulation/outputs/if/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runAcceleratorTest(nirFile, jsonFile)
  }

  test("IF with FIFO backpressure (size 128)") {
    val nirFile    = "../1-internal-simulation/outputs/if/model.nir"
    val jsonString = "../1-internal-simulation/outputs/if/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runBackpressureTest(nirFile, jsonFile, fifoSize = 128, almostFullThreshold = 100)
  }

  /**
   * Test accelerator behavior with simulated downstream FIFO backpressure.
   *
   * Simulates a FIFO of given size that:
   * - Accepts data when not full
   * - Asserts output_fifo_almost_full when threshold reached
   * - Stops accepting (TREADY=0) when full
   * - Drains when TLAST is seen
   */
  def runBackpressureTest(
    nirFile: String,
    jsonFile: String,
    fifoSize: Int,
    almostFullThreshold: Int
  ): Unit = {
    val config = ConfigJSON.fromJson(jsonFile)

    val timeout = 2000000 // Longer timeout for backpressure test
    SimConfig.withWave.withVerilator
      .workspacePath("./simWorkspace")
      .compile(new AcceleratorAXI(nirFile, jsonFile))
      .doSim { dut =>
        SimTimeout(timeout)
        dut.clockDomain.forkStimulus(2)
        sleep(10)
        initAxiLiteIdle(dut)

        // Use pre-generated packets from Python IOManager (optional for backward compatibility)
        val inputAXI: List[Long] = config.inputPackets match {
          case Some(packets) => packets
          case None          => List()
        }
        if (inputAXI.nonEmpty) {
          println(s"✓ Loaded ${inputAXI.length} pre-generated packets from ConfigJSON")
        } else {
          println("⚠ Warning: No input packets in ConfigJSON")
        }

        val inputQueue = scala.collection.mutable.Queue(inputAXI: _*)

        // Simulated FIFO state
        var fifoLevel      = 0
        var fifoAlmostFull = false

        // Output tracking
        val outputs           = scala.collection.mutable.ArrayBuffer[Int]()
        var timestepsRcvd     = 0
        var lastSeen          = false
        var tlastCount        = 0
        var tlastOnSpikeCount = 0 // Should remain 0 with timestep-based TLAST framing

        // Track expected input timesteps
        var expectedInputTimesteps = 0

        // Initialize signals
        // dut.io.output_fifo_almost_full #= false
        dut.io.m_axis.ready #= true

        // Monitor outputs and simulate FIFO behavior
        StreamMonitor(dut.io.m_axis, dut.clockDomain) { payload =>
          val packet          = payload.data.toInt
          val instructionType = packet & 0x7
          outputs += packet

          // Count ALL TIMESTEP packets (not just ones with TLAST)
          if (instructionType == TYPE_TIMESTEP) {
            timestepsRcvd += 1
            if (timestepsRcvd % 100 == 0 || timestepsRcvd <= 5) {
              println(s"  [TIMESTEP] Received timestep ${timestepsRcvd} (fifo level: ${fifoLevel + 1})")
            }
            if (timestepsRcvd >= config.timesteps) {
              lastSeen = true
            }
          }

          // Simulate FIFO filling
          fifoLevel += 1

          // Update almost_full signal (continuous check, not edge-triggered)
          val wasAlmostFull = fifoAlmostFull
          fifoAlmostFull = fifoLevel >= almostFullThreshold
          if (fifoAlmostFull && !wasAlmostFull) {
            println(s"  [FIFO] Almost full at level ${fifoLevel} (threshold ${almostFullThreshold})")
          }

          // Immediately update the signal
          // dut.io.output_fifo_almost_full #= fifoAlmostFull

          if (payload.last.toBoolean) {
            tlastCount += 1

            // Track whether TLAST was on a spike vs timestep
            if (instructionType == TYPE_SPIKE) {
              tlastOnSpikeCount += 1
              println(s"✗ Unexpected TLAST on SPIKE packet (count #${tlastOnSpikeCount}, fifo level: ${fifoLevel})")
            } else if (instructionType == TYPE_TIMESTEP) {
              println(s"✓ TLAST on TIMESTEP packet (fifo level: ${fifoLevel})")
            }

            // Simulate FIFO drain on TLAST
            println(s"  [FIFO] Draining: ${fifoLevel} -> 0")
            fifoLevel = 0
            fifoAlmostFull = false
            // dut.io.output_fifo_almost_full #= false
          }

          // Check if FIFO is full - stop accepting
          if (fifoLevel >= fifoSize) {
            println(s"  [FIFO] FULL at ${fifoLevel} - asserting backpressure (DEADLOCK if no TLAST!)")
          }
        }

        // Fork a process to continuously update m_axis.ready based on FIFO level
        // This is needed because StreamMonitor only fires on valid transactions,
        // so we can't update ready inside the callback after draining
        fork {
          while (!lastSeen) {
            dut.clockDomain.waitSampling()
            dut.io.m_axis.ready #= (fifoLevel < fifoSize)
          }
        }

        // Drive input packets to s_axis
        StreamDriver(dut.io.s_axis, dut.clockDomain) { payload =>
          if (inputQueue.nonEmpty) {
            val packet = inputQueue.dequeue()
            payload.data #= packet
            payload.last #= inputQueue.isEmpty
            payload.keep #= 0xf

            val instructionType = packet & 0x7
            if (instructionType == TYPE_TIMESTEP) {
              expectedInputTimesteps += 1
            }
            true
          } else {
            false
          }
        }

        // Wait for completion
        dut.clockDomain.waitSamplingWhere(timeout)(lastSeen)
        sleep(10)

        // Validate results
        val debugInputTimesteps  = dut.io.debug_input_timesteps.toBigInt
        val debugOutputTimesteps = dut.io.debug_output_timesteps.toBigInt

        println(s"\n=== Backpressure Test Results ===")
        println(s"✓ Test complete: Received ${outputs.length} output packets")
        println(s"✓ TLAST count: ${tlastCount} total (${tlastOnSpikeCount} on spikes, ${timestepsRcvd} on timesteps)")
        println(s"✓ Debug counters: input=${debugInputTimesteps}, output=${debugOutputTimesteps}")
        println(s"✓ Expected: input=${expectedInputTimesteps}, output=${timestepsRcvd}")

        // Assertions
        withClue(s"Backpressure: input timestep counter after ${outputs.length} packets: ") {
          debugInputTimesteps.toInt shouldBe expectedInputTimesteps
        }
        withClue(s"Backpressure: output timestep counter after ${outputs.length} packets: ") {
          debugOutputTimesteps.toInt shouldBe timestepsRcvd
        }
        withClue(s"Backpressure: total timesteps received: ") {
          timestepsRcvd shouldBe config.timesteps
        }

        withClue("Backpressure: TLAST on spike packets should be zero with timestep framing: ") {
          tlastOnSpikeCount shouldBe 0
        }
        withClue("Backpressure: one TLAST per emitted timestep expected: ") {
          tlastCount shouldBe timestepsRcvd
        }
        println(s"✓ Backpressure handling validated: TLAST aligned to timestep packets")
        println(s"✓ Backpressure test PASSED!")
      }
  }

  test("Linear -> IF") {
    val nirFile    = "../1-internal-simulation/outputs/linear-if/model.nir"
    val jsonString = "../1-internal-simulation/outputs/linear-if/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runAcceleratorTest(nirFile, jsonFile)
  }

  test("Linear -> LIF") {
    val nirFile    = "../1-internal-simulation/outputs/linear-lif/model.nir"
    val jsonString = "../1-internal-simulation/outputs/linear-lif/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runAcceleratorTest(nirFile, jsonFile)
  }

  test("Spiker MNIST") {
    val nirFile    = "../1-internal-simulation/outputs/spiker-mnist/model.nir"
    val jsonString = "../1-internal-simulation/outputs/spiker-mnist/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runAcceleratorTest(nirFile, jsonFile)
  }

  test("Spiker SHD") {
    val nirFile    = "../1-internal-simulation/outputs/spiker-shd/model.nir"
    val jsonString = "../1-internal-simulation/outputs/spiker-shd/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runAcceleratorTest(nirFile, jsonFile)
  }

  /**
   * Validates the runtime packet contract expected by PYNQ receiver logic:
   * - Exactly one TIMESTEP packet per timestep.
   * - TLAST asserted only on TIMESTEP packets (timestep framing).
   * - Spike coordinate always within [0, outputNeurons).
   * - Total spike packets bounded by timesteps * outputNeurons.
   */
  def runPacketContractTest(nirFile: String, jsonFile: String): Unit = {
    val config   = ConfigJSON.fromJson(jsonFile)
    val modelDir = nirFile.stripSuffix("/model.nir")

    // Recover output neuron count from recordings.npy shape [timesteps, outputNeurons]
    val outputNeurons = SimUtils.loadRecordingsNpy(s"$modelDir/recordings.npy").length / config.timesteps

    val timeout = 100000000
    SimConfig.withVerilator
      .workspacePath("./simWorkspace")
      .compile(new AcceleratorAXI(nirFile, jsonFile))
      .doSim { dut =>
        SimTimeout(timeout)
        dut.clockDomain.forkStimulus(2)
        sleep(10)
        initAxiLiteIdle(dut)

        val inputAXI: List[Long] = SimUtils.loadPacketsFromNpy(s"$modelDir/input_packets.npy")
        val inputQueue           = scala.collection.mutable.Queue(inputAXI: _*)

        var receivedTimesteps = 0
        var tlastCount        = 0
        var tlastOnSpikeCount = 0
        var spikeCount        = 0
        var maxSpikeCoord     = -1
        var done              = false

        dut.io.m_axis.ready #= true
        StreamMonitor(dut.io.m_axis, dut.clockDomain) { payload =>
          val packet          = payload.data.toInt
          val instructionType = packet & 0x7

          if (instructionType == TYPE_TIMESTEP) {
            receivedTimesteps += 1
            if (receivedTimesteps >= config.timesteps) {
              done = true
            }
          }

          if (instructionType == TYPE_SPIKE) {
            spikeCount += 1
            val coord = (packet >> 3) & 0x1fff
            if (coord > maxSpikeCoord) {
              maxSpikeCoord = coord
            }
          }

          if (payload.last.toBoolean) {
            tlastCount += 1
            if (instructionType == TYPE_SPIKE) {
              tlastOnSpikeCount += 1
            }
          }
        }

        StreamDriver(dut.io.s_axis, dut.clockDomain) { payload =>
          if (inputQueue.nonEmpty) {
            val packet = inputQueue.dequeue()
            payload.data #= packet
            payload.last #= inputQueue.isEmpty
            payload.keep #= 0xf
            true
          } else {
            false
          }
        }

        dut.clockDomain.waitSamplingWhere(timeout)(done)
        sleep(10)

        val maxAllowedSpikes = config.timesteps * outputNeurons

        withClue("Contract: TIMESTEP packet count should match config.timesteps: ") {
          receivedTimesteps shouldBe config.timesteps
        }
        withClue("Contract: TLAST should never be asserted on spike packets: ") {
          tlastOnSpikeCount shouldBe 0
        }
        withClue("Contract: one TLAST per timestep packet expected: ") {
          tlastCount shouldBe config.timesteps
        }
        withClue(s"Contract: spike coordinate should be < outputNeurons ($outputNeurons): ") {
          maxSpikeCoord should be < outputNeurons
        }
        withClue(s"Contract: total spikes should be <= timesteps * outputNeurons ($maxAllowedSpikes): ") {
          spikeCount should be <= maxAllowedSpikes
        }
      }
  }

  test("Packet stream/model contract (lif_norse)") {
    val nirFile    = "../1-internal-simulation/outputs/lif_norse/model.nir"
    val jsonString = "../1-internal-simulation/outputs/lif_norse/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runPacketContractTest(nirFile, jsonFile)
  }

  // ============================================================================
  // OUTPUT-CHECK TESTS - Compare hardware outputs against Python recordings
  // ============================================================================

  /**
   * Run an output-check test: drives AXI input packets, monitors the accelerator's
   * internal output stream (pre-AXI encoding), and compares against expected
   * recordings from the JSON config.
   *
   * Uses iverilog for faster compile time.
   */
  def runOutputCheckTest(
    nirFile: String,
    jsonFile: String,
    accelConfig: AcceleratorConfig = AcceleratorConfig.default,
    precision: Option[Int] = None
  ): Unit = {
    val config         = ConfigJSON.fromJson(jsonFile)
    val modelDir       = nirFile.stripSuffix("/model.nir")
    val recFlat        = SimUtils.loadRecordingsNpy(s"$modelDir/recordings.npy")
    val outputSize     = recFlat.length / config.timesteps
    val outputFracBits = config.quantizations("output")("input").qformat.fraction
    val outputPrec     = precision.getOrElse(outputFracBits)

    val timeout = 100000000
    SimConfig.withIVerilog
      .workspacePath("./simWorkspace")
      .compile(new AcceleratorAXI(nirFile, jsonFile, accelConfig))
      .doSim { dut =>
        SimTimeout(timeout)
        dut.clockDomain.forkStimulus(2)
        sleep(10)
        initAxiLiteIdle(dut)

        val inputAXI: List[Long] = SimUtils.loadPacketsFromNpy(s"$modelDir/input_packets.npy")
        val inputQueue           = scala.collection.mutable.Queue(inputAXI: _*)

        println(s"OutputCheck: ${inputAXI.length} packets, ${config.timesteps} timesteps, outputSize=$outputSize")

        // Accept AXI output (not checked here — we monitor the internal stream)
        dut.io.m_axis.ready #= true

        // Monitor accelerator.io.output (pre-AXI stream)
        val hwOutputs  = Array.fill(config.timesteps)(scala.collection.mutable.ArrayBuffer[(Int, Int)]())
        var hwTimestep = 0

        StreamMonitor(dut.accelerator.io.output, dut.clockDomain) { payload =>
          val coord = payload.fragment.coords(0).toInt
          val q     = SimUtils.quantizeToPrecision(payload.fragment.value(0).toDouble, outputPrec)
          if (q != 0 && hwTimestep < config.timesteps)
            hwOutputs(hwTimestep) += ((coord, q))
          if (payload.last.toBoolean)
            hwTimestep += 1
        }

        // Drive input packets
        StreamDriver(dut.io.s_axis, dut.clockDomain) { payload =>
          if (inputQueue.nonEmpty) {
            val packet = inputQueue.dequeue()
            payload.data #= packet
            payload.last #= inputQueue.isEmpty
            payload.keep #= 0xf
            true
          } else {
            false
          }
        }.setFactor(1.0f)

        // Wait for all timesteps
        dut.clockDomain.waitSamplingWhere(timeout)(hwTimestep >= config.timesteps)
        sleep(10)

        // Build expected output from recordings.npy at the requested precision
        val expected = Array.tabulate(config.timesteps) { t =>
          (0 until outputSize).flatMap { n =>
            val q = SimUtils.quantizeToPrecision(recFlat(t * outputSize + n), outputPrec)
            if (q != 0) Some((n, q)) else None
          }.toSet
        }

        // Compare each timestep
        var mismatches = 0
        for (t <- 0 until config.timesteps) {
          val hwSet  = hwOutputs(t).toSet
          val expSet = expected(t)
          if (hwSet != expSet) {
            mismatches += 1
            if (mismatches <= 5) {
              println(s"MISMATCH at timestep $t:")
              println(s"  HW  : $hwSet")
              println(s"  JSON: $expSet")
              val hwOnly  = hwSet -- expSet
              val expOnly = expSet -- hwSet
              if (hwOnly.nonEmpty) println(s"  HW-only:   $hwOnly")
              if (expOnly.nonEmpty) println(s"  JSON-only: $expOnly")
            }
          }
        }

        withClue(
          s"OutputCheck: $mismatches/${config.timesteps} timesteps mismatched " +
            s"(first 5 shown above): "
        ) {
          mismatches shouldBe 0
        }

        println(s"OutputCheck PASSED: all ${config.timesteps} timesteps match recordings")
      }
  }

  def runAxiLiteCounterTrackingTest(nirFile: String, jsonFile: String): Unit = {
    val config = ConfigJSON.fromJson(jsonFile)

    val timeout = 100000000
    SimConfig.withIVerilog
      .workspacePath("./simWorkspace")
      .compile(new AcceleratorAXI(nirFile, jsonFile))
      .doSim { dut =>
        SimTimeout(timeout)
        dut.clockDomain.forkStimulus(2)
        sleep(10)
        initAxiLiteIdle(dut)

        val inputAXI: List[Long] = config.inputPackets match {
          case Some(packets) => packets
          case None          => List()
        }
        val inputQueue = scala.collection.mutable.Queue(inputAXI: _*)

        var expectedInputTimesteps  = 0
        var expectedOutputTimesteps = 0
        var completed               = false
        var pollViolations          = 0

        dut.io.m_axis.ready #= true

        // Randomized backpressure to exercise AXI-Lite reads during active streaming.
        fork {
          while (!completed) {
            dut.clockDomain.waitSampling()
            dut.io.m_axis.ready #= (scala.util.Random.nextInt(100) < 70)
          }
          dut.io.m_axis.ready #= true
        }

        fork {
          while (!completed) {
            dut.clockDomain.waitSampling()

            if (dut.io.s_axis.valid.toBoolean && dut.io.s_axis.ready.toBoolean) {
              val packetType = (dut.io.s_axis.data.toBigInt & 0x7).toInt
              if (packetType == TYPE_TIMESTEP) {
                expectedInputTimesteps += 1
              }
            }

            if (dut.io.m_axis.valid.toBoolean && dut.io.m_axis.ready.toBoolean) {
              val packetType = (dut.io.m_axis.data.toBigInt & 0x7).toInt
              if (packetType == TYPE_TIMESTEP) {
                expectedOutputTimesteps += 1
                if (expectedOutputTimesteps >= config.timesteps) {
                  completed = true
                }
              }
            }
          }
        }

        StreamDriver(dut.io.s_axis, dut.clockDomain) { payload =>
          if (inputQueue.nonEmpty) {
            val packet = inputQueue.dequeue()
            payload.data #= packet
            payload.last #= inputQueue.isEmpty
            payload.keep #= 0xf
            true
          } else {
            false
          }
        }

        // Poll AXI-Lite registers while traffic is active and verify monotonic behavior.
        var lastInRead  = BigInt(0)
        var lastOutRead = BigInt(0)

        fork {
          while (!completed) {
            dut.clockDomain.waitSampling(25)
            val inRead  = axiLiteRead(dut, AXIL_INPUT_TIMESTEPS_ADDR)
            val outRead = axiLiteRead(dut, AXIL_OUTPUT_TIMESTEPS_ADDR)

            if (inRead < lastInRead) {
              pollViolations += 1
            }
            if (outRead < lastOutRead) {
              pollViolations += 1
            }
            if (inRead.toInt > expectedInputTimesteps) {
              pollViolations += 1
            }
            if (outRead.toInt > expectedOutputTimesteps) {
              pollViolations += 1
            }

            lastInRead = inRead
            lastOutRead = outRead
          }
        }

        dut.clockDomain.waitSamplingWhere(timeout)(completed)
        sleep(10)

        val finalInRead   = axiLiteRead(dut, AXIL_INPUT_TIMESTEPS_ADDR)
        val finalOutRead  = axiLiteRead(dut, AXIL_OUTPUT_TIMESTEPS_ADDR)
        val finalInDebug  = dut.io.debug_input_timesteps.toBigInt
        val finalOutDebug = dut.io.debug_output_timesteps.toBigInt

        withClue("AXI-Lite input timestep register should equal observed stream events: ") {
          finalInRead.toInt shouldBe expectedInputTimesteps
        }
        withClue("AXI-Lite output timestep register should equal observed stream events: ") {
          finalOutRead.toInt shouldBe expectedOutputTimesteps
        }
        withClue("AXI-Lite input timestep register should match debug wire: ") {
          finalInRead shouldBe finalInDebug
        }
        withClue("AXI-Lite output timestep register should match debug wire: ") {
          finalOutRead shouldBe finalOutDebug
        }
        withClue("AXI-Lite polled values should stay monotonic and bounded during streaming: ") {
          pollViolations shouldBe 0
        }
      }
  }

  /**
   * Convenience wrapper: run an output-check test by model directory name.
   * Constructs paths as ../1-internal-simulation/outputs/<modelName>/model.{nir,json}.
   */
  def runOutputCheckForModel(
    modelName: String,
    accelConfig: AcceleratorConfig = AcceleratorConfig.default
  ): Unit = {
    val basePath = s"../1-internal-simulation/outputs/$modelName"
    val nirFile  = basePath + "/model.nir"
    val jsonFile = scala.io.Source.fromFile(basePath + "/model.json").mkString
    runOutputCheckTest(nirFile, jsonFile, accelConfig)
  }

  test("OutputCheck: Linear16 -> LIF8 -> Linear4") {
    val nirFile    = "../1-internal-simulation/outputs/linear16-lif8-linear4/model.nir"
    val jsonString = "../1-internal-simulation/outputs/linear16-lif8-linear4/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runOutputCheckTest(nirFile, jsonFile)
  }

  test("OutputCheck: Linear -> LIF -> Linear") {
    val nirFile    = "../1-internal-simulation/outputs/linear-lif-linear/model.nir"
    val jsonString = "../1-internal-simulation/outputs/linear-lif-linear/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runOutputCheckTest(nirFile, jsonFile)
  }

  test("OutputCheck: lif-16b") {
    runOutputCheckForModel("lif-16b")
  }

  test("AXI-Lite counters track stream events") {
    val nirFile    = "../1-internal-simulation/outputs/if/model.nir"
    val jsonString = "../1-internal-simulation/outputs/if/model.json"
    val jsonFile   = scala.io.Source.fromFile(jsonString).mkString
    runAxiLiteCounterTrackingTest(nirFile, jsonFile)
  }
}
