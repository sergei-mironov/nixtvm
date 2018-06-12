{ pkgs ?  import <nixpkgs> {}
, stdenv ? pkgs.stdenv
} :

let
  inherit (pkgs) fetchgit fetchgitLocal;
  inherit (builtins) filterSource;
  inherit (pkgs.lib.sources) cleanSourceFilter;

  pp = pkgs.python36Packages;

  tvm_cmake_flags = "-DINSTALL_DEV=ON -DUSE_CUDA=On";
in

rec {

  tvm = stdenv.mkDerivation rec {
    name = "tvm";
    src = ../tvm-clean;
    buildInputs = with pkgs; [cmake];

    cmakeFlags = tvm_cmake_flags;
  };

  tvm-python = pp.buildPythonPackage rec {
    pname = "tvm";
    version = "0.8";
    name = "${pname}-${version}";
    src = ../tvm-clean/python;
    buildInputs = with pkgs; with pp; [
      tvm decorator numpy tornado cudatoolkit
    ];

    preConfigure = ''
      export LD_LIBRARY_PATH="${tvm}/lib:$LD_LIBRARY_PATH";
    '';
  };

  tvm-python-topi = pp.buildPythonPackage rec {
    pname = "tvm";
    version = "0.8";
    name = "${pname}-${version}";
    src = ../tvm-clean;
    buildInputs = with pkgs; with pp; [
      tvm tvm-python decorator numpy tornado
      scipy nose
    ];

    preConfigure = ''
      cd topi/python
      export LD_LIBRARY_PATH="${tvm}/lib:$LD_LIBRARY_PATH";
    '';

    doCheck=false;
  };

  nnvm = stdenv.mkDerivation {
    name = "nnvm";

    # src = filterSource (path: type :
    #      (cleanSourceFilter path type)
    #   && !(baseNameOf path == "build" && type == "directory")
    #   && !(baseNameOf path == "lib" && type == "directory")
    #   ) ../nnvm;
    src = ../tvm-clean;

    cmakeFlags = "-DBUILD_STATIC_NNVM=On";

    buildInputs = with pkgs; with pp; [
      cmake
      python
      setuptools
      gfortran
    ];


  # installPhase = ''
  #   mkdir -pv $out/lib
  #   cp lib/* $out/lib
  # '';
  };


  shell = stdenv.mkDerivation {
    name = "shell";

    buildInputs = with pkgs; with pp; [
      cmake decorator numpy tornado pp.nose ncurses zlib scipy
    ];

    inherit tvm_cmake_flags;

    shellHook = ''
      TVM=$HOME/proj/tvm
      export PYTHONPATH="$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"
      export LD_LIBRARY_PATH="$TVM:$LD_LIBRARY_PATH"
      cd $TVM
    '';
  };




  # nnvm-python = pp.buildPythonPackage rec {
  #   pname = "nnvm";
  #   version = "0.8";
  #   name = "${pname}-${version}";

  #   src = ../nnvm/python;


  #   preConfigure = ''
  #     export LD_LIBRARY_PATH="${nnvm}/lib:${tvm}/lib:$LD_LIBRARY_PATH";
  #   '';

  #   buildInputs = with pp; [
  #     numpy
  #     tvm
  #     tvm-python
  #     tvm-python-topi
  #     decorator
  #     pillow
  #     opencv3
  #     cffi
  #   ];


  # # cd python
  # # export PYTHONPATH=`pwd`:$PYTHONPATH
  # # python setup.py install --prefix=$out
  # # cd ..
  # };
}

