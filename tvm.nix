{ pkgs ?  import <nixpkgs> {}
, stdenv ? pkgs.stdenv
, tvmCmakeFlagsEx ? abort "Use tvm-<mode>.nix"
, tvmDepsEx ? abort "Use tvm-<mode>.nix"
} :

let
  inherit (pkgs) fetchgit fetchgitLocal;
  inherit (builtins) filterSource;
  inherit (pkgs.lib.sources) cleanSourceFilter;

  pp = pkgs.python36Packages;

  tvmCmakeFlags = "-DINSTALL_DEV=ON " + tvmCmakeFlagsEx;
  tvmDeps = [ pp.pillow ] ++ tvmDepsEx;
in

rec {

  tvm = stdenv.mkDerivation rec {
    name = "tvm";
    src = ../tvm-clean;
    buildInputs = with pkgs; [cmake] ++ tvmDeps;
    cmakeFlags = tvmCmakeFlags;
  };

  tvm-python = pp.buildPythonPackage rec {
    pname = "tvm";
    version = "0.8";
    name = "${pname}-${version}";
    src = ../tvm-clean/python;
    buildInputs = with pkgs; with pp; [tvm decorator numpy tornado];

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


  mktags = pkgs.writeShellScriptBin "mktags.sh" ''
    find -name '*cc' -or -name '*hpp' -or -name '*h' | \
      ctags -L - --c++-kinds=+p --fields=+iaS --extra=+q --language-force=C++
    find -name '*py' | \
      ctags --append -L -
  '';


  shell = stdenv.mkDerivation {
    name = "shell";

    buildInputs = with pkgs; with pp; [
      cmake decorator numpy tornado pp.nose ncurses zlib scipy mktags
      mxnet scikitlearn numpy matplotlib
    ] ++ tvmDeps;

    inherit tvmCmakeFlags;

    shellHook = ''
      if test -f /etc/myprofile ; then
        . /etc/myprofile
      fi

      TVM=$HOME/proj/tvm
      export PYTHONPATH="$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"
      # export LD_LIBRARY_PATH="$TVM:$LD_LIBRARY_PATH"
      cd $TVM
    '';
  };

}

