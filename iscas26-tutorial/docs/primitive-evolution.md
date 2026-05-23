# Developing the NIR LIF primitive, starting from the I.

# What this is

A live-coding script for growing one hardware neuron primitive through
three NIR node types, in increasing order of biological detail:

1. **I**   -- Integrate.               v[t] = v[t-1] + input[t]
2. **LI**  -- Leaky Integrate.          v[t] = alpha*v[t-1] + input[t]
3. **LIF** -- Leaky Integrate-and-Fire. v[t] = alpha*v[t-1] + (1-alpha)*input[t],
            emit a spike and reset when v[t] >= v_threshold.

Each step is a small, self-contained edit to the SpinalHDL primitive in
`iscas26-tutorial/primitive_implementations/`. The point of the exercise
is **not** the neuron maths -- it is to see how a NIR node type is wired
into NIR2FPGA: a parameter mapping (`types/`), a dispatch entry
(`PrimitiveHW.scala`), and a hardware implementation (`impl/`).

# The codebase: three files per primitive

```
iscas26-tutorial/primitive_implementations/
  PrimitiveHW.scala     <- dispatch: NIR params -> *HW builder
  types/i.scala         <- IParams   -> Neuron.Config   (parameter mapping)
  types/li.scala        <- LIParams  -> Neuron.Config
  types/lif.scala       <- LIFParams -> Neuron.Config
  impl/Neuron.scala     <- the actual hardware (a streaming datapath)
  impl/Affine.scala     <- the Linear/Affine layer (unchanged here)
```

The dataflow when NIR2FPGA compiles a graph:

```
NIR node (e.g. nir.LIF)
    -> PrimitiveHW.create  matches the *Params type
    -> LIFHW.makeHardware   maps NIR params into a Neuron.Config
    -> Neuron(config)       elaborates the SpinalHDL datapath
```

`Neuron.scala` is **one unified component**. Its behaviour is selected by
which `Config` fields are populated:

| Config has...                    | Behaves as |
|----------------------------------|------------|
| (no tau)                         | I          |
| tau                              | LI         |
| tau + v_reset + v_threshold      | LIF        |

# PART 1 -- I -> LI : add the leak

## Concept

A plain integrator never forgets: every input current is added to the
membrane forever. A **leaky** integrator multiplies the membrane by a
decay factor `alpha` each timestep, so old input fades away:

v[t] = alpha * v[t-1] + input[t]

`alpha` comes from the membrane time constant `tau`. We use the simple
discrete approximation `alpha = 1 - 2/tau`. With `tau = None` we keep
`alpha = 1`, i.e. no leak -- so the same code still serves the I node.

## Step 1.1 -- `impl/Neuron.scala` : add the leak stage

FIND the end of the `memReadWithPayload` block and the `// Integrate`
comment that follows it:

```scala
  val memReadWithPayload = mem.streamReadSync(
    input.translateWith(addr),
    input.payload
  )

  // Integrate: v[t] = v[t-1] + input[t]
```

REPLACE with (i.e. insert the leak stage between them):

```scala
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
```

`leaked` is a new pipeline stage: it reads the membrane `mem` and scales
it by `alpha`. It carries the input payload through untouched (the
`TupleBundle`) so the next stage can still see it.

## Step 1.2 -- `impl/Neuron.scala` : feed `integrated` from `leaked`

`integrated` currently reads `memReadWithPayload` directly. Re-point it
at the new `leaked` stage. (The tuple field names change: `leaked`
emits `(leakedState, input)`, accessed as `_1` / `_2`.)

FIND the head of the `integrated` block:

```scala
  val integrated = memReadWithPayload.map { p =>
    val state = p.value
    val input = p.linked

    val integratedState = Vec(state.zip(input.fragment.value).map { case (v, inp) =>
```

REPLACE with:

```scala
  val integrated = leaked.map { p =>
    val leakedState = p._1
    val input       = p._2

    val integratedState = Vec(leakedState.zip(input.fragment.value).map { case (v, inp) =>
```

Nothing after `integrated` changes: `mem.write` and the output stage
still read `integrated`.

## Step 1.5 -- `PrimitiveHW.scala` : dispatch `LIParams`

FIND the `params match` block in `PrimitiveHW.create`:

```scala
      case p: IParams      => IHW(id, p, config)
      case p: AffineParams => AffineHW(id, p, config)
```

REPLACE with:

```scala
      case p: IParams      => IHW(id, p, config)
      case p: LIParams     => LIHW(id, p, config)
      case p: AffineParams => AffineHW(id, p, config)
```

## Checkpoint -- Part 1

```bash
cd 2-compilation
sbt -DprimitivesDir=$(pwd)/../iscas26-tutorial/primitive_implementations compile
```

Expect `done compiling` / `success`. The codebase is now equivalent to
the `main` branch (Linear + I + LI). This is where attendees begin.

# PART 2 -- LI -> LIF : add fire-and-reset

> **Audience:** presenter + attendees (main branch)

## Concept

A leaky integrator only ever produces a continuous membrane voltage. A
**spiking** neuron adds a threshold: when `v[t]` crosses `v_threshold` it

1. emits a spike (output becomes 1 instead of the voltage), and
2. resets the membrane to 0.

LIF also **normalises** the input by `(1 - alpha)`, so the membrane has a
bounded steady state regardless of `tau`:

```
v[t] = alpha * v[t-1] + (1 - alpha) * input[t]
```

We add one more datapath stage, `afterFiring`, after `integrated`.

## Step 2.2 -- `impl/Neuron.scala` : the `isLIF` flag and input scaling

FIND the `memReadWithPayload` block through to `alphaFactor`:

```scala
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
```

REPLACE with:

```scala
  val memReadWithPayload = mem.streamReadSync(
    input.translateWith(addr),
    input.payload
  )

  val isLIF = c.v_threshold.isDefined && c.v_reset.isDefined

  // Leak factor: alpha = 1 - 2/tau. With tau = None, alpha = 1 (no leak),
  // which makes this neuron behave as a plain integrator (I).
  val alpha = c.tau match {
    case Some(tau) => 1.0 - 2.0 / tau
    case None      => 1.0
  }

  // LIF normalises the input by (1 - alpha) for a bounded steady state.
  // I and LI leave the input unscaled (inputScale = 1).
  val inputScale = if (isLIF) 1.0 - alpha else 1.0
  val rFactor    = AF(inputScale, c.quants("v_mem").qformat)

  val alphaFactor = Vec(AF(alpha, c.quants("v_mem").qformat), c.input.width)
```

## Step 2.3 -- `impl/Neuron.scala` : scale the input in `integrated`

FIND the whole `integrated` block:

```scala
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
```

REPLACE with:

```scala
  // Integrate: v[t] = alpha * v[t-1] + (1 - alpha) * input[t]
  val integrated = leaked.map { p =>
    val leakedState = p._1
    val input       = p._2

    val integratedState = Vec(leakedState.zip(input.fragment.value).map { case (v, inp) =>
      (v + (rFactor * inp)
        .fixTo(c.quants("v_mem").qformat, RoundType.CEIL))
        .fixTo(c.quants("v_mem").qformat, RoundType.CEIL)
    })

    TupleBundle(integratedState, input)
  }
```

For I and LI, `rFactor` is 1.0, so this is still just `v + input`.

## Step 2.4 -- `impl/Neuron.scala` : the `afterFiring` stage

This is the spike. FIND the tail of `integrated` and the `mem.write`
that follows it:

```scala
    TupleBundle(integratedState, input)
  }

  // Write updated membrane back
  mem.write(
    address = integrated.payload._2.fragment.flattenedAddress,
    data = integrated.payload._1,
    enable = integrated.fire
  )
```

REPLACE with (insert `afterFiring`, and re-point `mem.write` at it):

```scala
    TupleBundle(integratedState, input)
  }

  val afterFiring = (c.v_threshold, c.v_reset) match {
    case (Some(vth), Some(vreset)) =>
      integrated.map { p =>
        val integratedState = p._1
        val input           = p._2

        val vthFactor = AF(vth, c.quants("v_mem").qformat)
        val fired     = Vec(integratedState.map(_ >= vthFactor))

        val nextState = Vec(integratedState.zip(fired).map { case (v, f) =>
          Mux(f, AF(0.0, c.quants("v_mem").qformat), v)
        })

        TupleBundle(nextState, input, fired)
      }
    case _ =>
      integrated.map(p => TupleBundle(p._1, p._2, Vec(Seq.fill(c.input.width)(False))))
  }

  // Write updated membrane back
  mem.write(
    address = afterFiring.payload._2.fragment.flattenedAddress,
    data = afterFiring.payload._1,
    enable = afterFiring.fire
  )
```

`afterFiring` compares the membrane to `v_threshold`, resets fired
neurons to 0, and carries a third tuple field -- the `fired` bit Vec --
through to the output. When `v_threshold` / `v_reset` are absent (I and
LI) the `case _` branch passes the membrane through with all-False
spikes, so `afterFiring` is harmless for the non-spiking nodes.

Note `mem.write` now stores `afterFiring` (the **reset** membrane), not
the raw integrated value.

## Step 2.5 -- `impl/Neuron.scala` : emit spikes on the output

FIND the output stage:

```scala
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
```

REPLACE with:

```scala
  // --- Drive output ---
  afterFiring.translateInto(output) { case (out, from) =>
    val state  = from._1
    val input  = from._2
    val spikes = from._3
    out.last            := input.last
    out.fragment.coords := input.fragment.coords
    out.fragment.value := Vec(
      if (isLIF)
        spikes.map(s => Mux(s, AF(1.0, c.quants("output").qformat), AF(0.0, c.quants("output").qformat)))
      else
        state.map(_.fixTo(c.quants("output").qformat, RoundType.FLOOR))
    )
  }
```

The output now reads `afterFiring` and unpacks the `spikes` field
(`_3`). For a LIF node it emits 1/0 spikes; for I and LI (`isLIF` false)
it still emits the membrane voltage.

`Neuron.scala` is now complete -- one component that serves I, LI and
LIF.

## Step 2.7 -- `PrimitiveHW.scala` : dispatch `LIFParams`

FIND the `params match` block:

```scala
      case p: IParams      => IHW(id, p, config)
      case p: LIParams     => LIHW(id, p, config)
      case p: AffineParams => AffineHW(id, p, config)
```

REPLACE with:

```scala
      case p: IParams      => IHW(id, p, config)
      case p: LIParams     => LIHW(id, p, config)
      case p: LIFParams    => LIFHW(id, p, config)
      case p: AffineParams => AffineHW(id, p, config)
```

## Checkpoint -- Part 2

```bash
cd 2-compilation
sbt -DprimitivesDir=$(pwd)/../iscas26-tutorial/primitive_implementations compile
```

Then the real finish line -- the classifier notebook now has every
primitive it needs:

```bash
cd iscas26-tutorial/neuron
jupytext --to notebook --execute 2-n2f.py
```

`2-n2f.py` should compile `Linear -> LIF`, simulate it, and report an
accuracy matching the `solution` branch.

# Recap

| Step | File              | What you added                                   |
|------|-------------------|--------------------------------------------------|
| 1.1  | impl/Neuron.scala | `alpha`, `alphaFactor`, the `leaked` stage        |
| 1.2  | impl/Neuron.scala | `integrated` now consumes `leaked`                |
| 1.5  | PrimitiveHW.scala | `LIParams` dispatch case                          |
| 2.2  | impl/Neuron.scala | `isLIF`, `inputScale`, `rFactor`                  |
| 2.3  | impl/Neuron.scala | input scaled by `(1-alpha)` in `integrated`       |
| 2.4  | impl/Neuron.scala | the `afterFiring` stage; `mem.write` reads it     |
| 2.5  | impl/Neuron.scala | output emits spikes; reads `afterFiring`          |
| 2.7  | PrimitiveHW.scala | `LIFParams` dispatch case                         |

Adding a NIR primitive is always these three moves: a `types/` mapping,
a `PrimitiveHW` dispatch case, and whatever hardware the `impl/`
component needs.
