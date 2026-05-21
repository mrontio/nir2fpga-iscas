{ pkgs, lib, config, inputs, ... }:
let
  torchVariant = "cpu"; #cu124;
  python = pkgs.python311;
  pythonPackages = python.pkgs;
  nirFork = pythonPackages.buildPythonPackage {
    pname = "nir";
    version = "100.0.0";
    src = inputs.nir-fork;
    pyproject = true;
    build-system = with pythonPackages; [ setuptools setuptools-scm ];
    dependencies = with pythonPackages; [ numpy h5py ];
    doCheck = false;
    SETUPTOOLS_SCM_PRETEND_VERSION = "100.0.0";
  };
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
    CPATH = lib.makeIncludePath [
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
    venv = {
      enable = true;
      requirements = ''
        --extra-index-url https://download.pytorch.org/whl/${torchVariant}
        torch==2.10.0+${torchVariant}
        norse==1.1.0
        numpy==1.26.4
        matplotlib==3.10.8
        sympy==1.14.0
        tonic==1.6.0
        sinabs==3.1.3
        pandas==3.0.1
        seaborn==0.13.2
        vcdvcd==2.6.0
        setvcd==0.8.0
        tqdm==4.67.3
        typing_extensions==4.14.1
        pyright==1.1.403
        jax
        jaxlib
        dm-haiku
        optax
        spyx
        snntorch
        notebook
        ipywidgets
        jupytext==1.16.7
        -e ${config.devenv.root}/1-discretization-quantization/InternalSimulator
      '';
    };
  };

  git-hooks = {
    default_stages = [ "pre-commit" ];
    hooks = {
      scalafmt = {
        enable = true;
        description = "Format Scala code";
        entry = "${pkgs.scalafmt}/bin/scalafmt";
        files = "\\.scala$";
        language = "system";
      };
      pyright = {
        enable = true;
        description = "Run pyright type checker on InternalSimulator package";
        entry = "pyright --project ./pyrightconfig.json";
        files = "\\.py$";
        pass_filenames = false;
      };
    };
  };

  enterShell = ''
    export PYTHONPATH="${nirFork}/${python.sitePackages}:$PYTHONPATH"
    echo "Torch version: ${torchVariant} (change top of devenv.nix to change this)"
  '';

  enterTest = ''
  '';

}
