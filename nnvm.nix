{ pkgs ?  import <nixpkgs> {}
, stdenv ? pkgs.stdenv
} :

let
  inherit (builtins) filterSource;
  inherit (pkgs.lib.sources) cleanSourceFilter;

  pp = pkgs.python36Packages;
in

rec {
  nnvm = stdenv.mkDerivation {

    src = filterSource (path: type :
         (cleanSourceFilter path type)
      && !(baseNameOf path == "build" && type == "directory")
      && !(baseNameOf path == "lib" && type == "directory")
      ) ../nnvm;

    name = "nnvm";
    buildInputs = with pkgs; with pp; [
      python
      setuptools
      gfortran
    ];


  installPhase = ''
    mkdir -pv $out/lib
    cp lib/* $out/lib
  '';
  };

  tvm = stdenv.mkDerivation rec {
    name = "tvm";
    src = ../nnvm/tvm;
    buildInputs = [];
    installPhase = ''
      mkdir -pv $out/lib
      make installdev DESTDIR=$out
    '';
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
    ];

    # cd python
    # export PYTHONPATH=`pwd`:$PYTHONPATH
    # python setup.py install --prefix=$out
    # cd ..
  };
}

