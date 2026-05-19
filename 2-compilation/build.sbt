ThisBuild / version      := "1.0"
ThisBuild / scalaVersion := "2.13.15"
ThisBuild / organization := "org.example"

val spinalVersion    = "1.14.0"
val spinalCore       = "com.github.spinalhdl" %% "spinalhdl-core" % spinalVersion
val spinalLib        = "com.github.spinalhdl" %% "spinalhdl-lib"  % spinalVersion
val spinalIdslPlugin = compilerPlugin("com.github.spinalhdl" %% "spinalhdl-idsl-plugin" % spinalVersion)
val javaHDF          = "io.jhdf"               % "jhdf"           % "0.6.5"
val scalaTest        = "org.scalatest"        %% "scalatest"      % "3.2.17" % Test

// Safe JSON parsing
val circeVersion = "0.14.9"
val circeCore    = "io.circe" % s"circe-core_2.13"   % circeVersion
val circeGeneric = "io.circe" % "circe-generic_2.13" % circeVersion
val circeParser  = "io.circe" % "circe-parser_2.13"  % circeVersion

// Print full stack traces in tests
// They tend to be big with SpinalHDL
Test / testOptions += Tests.Argument("-oF")

lazy val nir = ProjectRef(file("./nir4s"), "nir")

// Optional custom primitives directory (absolute or relative to project root).
// Override with: sbt -DprimitivesDir=path/to/custom/primitives <task>
// Must follow the same structure as the default: PrimitiveHW.scala, types/, impl/
val customPrimitivesDir = sys.props.get("primitivesDir")

// Resolve to an absolute File so absolute paths work correctly.
val resolvedCustomDir: Option[java.io.File] = customPrimitivesDir.map { p =>
  val f = new java.io.File(p)
  if (f.isAbsolute) f else new java.io.File(System.getProperty("user.dir"), p)
}

// Only swap out the default if the custom directory actually exists on disk.
val useCustomPrimitives: Boolean = resolvedCustomDir.exists(_.isDirectory)

Global / onLoad := {
  val previous = (Global / onLoad).value
  (state: State) => {
    val primitivesMsg = (customPrimitivesDir, useCustomPrimitives) match {
      case (Some(p), true)  => s"custom: $p"
      case (Some(p), false) => s"default (WARNING: custom dir '$p' not found — falling back)"
      case _                => "default: hw/spinal/NIR2FPGA/src/primitives"
    }
    state.log.info(s"[NIR2FPGA] Primitives: $primitivesMsg")
    previous(state)
  }
}

lazy val NIR2FPGA = (project in file("."))
  .dependsOn(nir)
  .settings(
    name := "NIR2FPGA",
    Compile / scalaSource := baseDirectory.value / "hw" / "spinal",
    Compile / unmanagedSourceDirectories ++= resolvedCustomDir
      .filter(_ => useCustomPrimitives)
      .toSeq,
    Compile / unmanagedSources := {
      val base = baseDirectory.value
      val systemTestDir  = (base / "hw/spinal/NIR2FPGA/test").getAbsolutePath
      val defaultPrimDir = (base / "hw/spinal/NIR2FPGA/src/primitives").getAbsolutePath
      val defaultImplTest  = (base / "hw/spinal/NIR2FPGA/src/primitives/impl/test").getAbsolutePath
      val defaultTypesTest = (base / "hw/spinal/NIR2FPGA/src/primitives/types/test").getAbsolutePath
      val customImplTest  = resolvedCustomDir.filter(_ => useCustomPrimitives).map(_ / "impl/test").map(_.getAbsolutePath)
      val customTypesTest = resolvedCustomDir.filter(_ => useCustomPrimitives).map(_ / "types/test").map(_.getAbsolutePath)

      val excludeDirs = Seq(systemTestDir, defaultImplTest, defaultTypesTest) ++
        customImplTest ++ customTypesTest ++
        (if (useCustomPrimitives) Seq(defaultPrimDir) else Seq.empty)

      (Compile / unmanagedSources).value
        .filterNot(f => excludeDirs.exists(d => f.getAbsolutePath.startsWith(d)))
    },
    // System tests (Accelerator, AXI, etc.) always loaded from test/
    Test / scalaSource := baseDirectory.value / "hw" / "spinal" / "NIR2FPGA" / "test",
    // Primitive tests co-located with the active primitives directory
    Test / unmanagedSourceDirectories ++= {
      val primDir = resolvedCustomDir
        .filter(_ => useCustomPrimitives)
        .getOrElse(baseDirectory.value / "hw/spinal/NIR2FPGA/src/primitives")
      Seq(primDir / "impl/test", primDir / "types/test")
    },
    libraryDependencies ++= Seq(
      spinalCore,
      spinalLib,
      spinalIdslPlugin,
      javaHDF,
      scalaTest,
      circeCore,
      circeGeneric,
      circeParser
    )
  )

fork := true
