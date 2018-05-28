{ pkgs ?  import <nixpkgs> {}
, stdenv ? pkgs.stdenv
} :

let
  inherit (builtins) filterSource;
  inherit (pkgs.lib.sources) cleanSourceFilter;

  pp = pkgs.python36Packages;
in

rec {

  tvm = stdenv.mkDerivation rec {
    name = "tvm";
    src = ../nnvm/tvm;
    buildInputs = with pkgs; [cmake];

    cmakeFlags = "-DINSTALL_DEV=ON";
  };

  tvm-python = pp.buildPythonPackage rec {
    pname = "tvm";
    version = "0.8";
    name = "${pname}-${version}";
    src = ../nnvm/tvm/python;
    buildInputs = with pkgs; with pp; [tvm decorator numpy tornado];

    preConfigure = ''
      export LD_LIBRARY_PATH="${tvm}/lib:$LD_LIBRARY_PATH";
    '';
  };

  tvm-python-topi = pp.buildPythonPackage rec {
    pname = "tvm";
    version = "0.8";
    name = "${pname}-${version}";
    src = ../nnvm/tvm;
    buildInputs = with pkgs; with pp; [tvm tvm-python decorator numpy tornado];

    preConfigure = ''
      cd topi/python
      export LD_LIBRARY_PATH="${tvm}/lib:$LD_LIBRARY_PATH";
    '';

    doCheck=false;
  };

  nnvm = stdenv.mkDerivation {
    name = "nnvm";

    src = filterSource (path: type :
         (cleanSourceFilter path type)
      && !(baseNameOf path == "build" && type == "directory")
      && !(baseNameOf path == "lib" && type == "directory")
      ) ../nnvm;

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


  nnvm-python = pp.buildPythonPackage rec {
    pname = "nnvm";
    version = "0.8";
    name = "${pname}-${version}";

    src = ../nnvm/python;


    preConfigure = ''
      export LD_LIBRARY_PATH="${nnvm}/lib:${tvm}/lib:$LD_LIBRARY_PATH";
    '';

    buildInputs = with pp; [
      numpy
      tvm
      nnvm
      tvm-python
      tvm-python-topi
      decorator
      pillow
      opencv3
      cffi
    ];


  # cd python
  # export PYTHONPATH=`pwd`:$PYTHONPATH
  # python setup.py install --prefix=$out
  # cd ..
  };
}

