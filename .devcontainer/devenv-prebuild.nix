{ pkgs, lib, config, inputs, ... }:
# Mirrors devenv.nix but omits the Python venv requirements.
# Used only during Docker image build to pre-populate /nix/store.
# The full Python venv (torch, jax, etc.) is built at post-create time.
let
  python = pkgs.python311;
  pythonPackages = python.pkgs;
in
{
  env = {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.boost
      pkgs.glibc.dev
      pkgs.stdenv.cc.cc.lib
      pkgs.libgcc.lib
      pkgs.libGL
      pkgs.hdf5
      pkgs.zlib
    ];
    CPATH = lib.makeIncludePath 
      pkgs.boost
      pkgs.glibc.dev
    ];
    YOSYS_GHDL_EXTENSION = pkgs.yosys-ghdl.outPath + "/share/yosys/plugins/ghdl.so";
  };

  packages = [
    pkgs.git
    pkgs.git-lfs
    pkgs.stdenv.cc
    pkgs.boost
    pkgs.iverilog
    pkgs.ghdl
    pkgs.yosys
    pkgs.yosys-ghdl
    pkgs.verilator
    pkgs.surfer
    pkgs.metals
    pkgs.scalafmt
    pkgs.hdf5
  ];

  languages.java = {
    jdk.package = pkgs.jdk17;
  };

  languages.scala = {
    enable = true;
    package = pkgs.scala_2_13;
    sbt.enable = true;
  };

  languages.python = {
    enable = true;
    package = python;
  };
}
