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

    src = ../tvm-clean;

    cmakeFlags = "-DBUILD_STATIC_NNVM=On";

    buildInputs = with pkgs; with pp; [
      cmake
      python
      setuptools
      gfortran
    ];
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
      cmake
      decorator
      tornado
      pp.nose
      ncurses
      zlib
      scipy
      mktags
      numpy
      scikitlearn
      matplotlib
      ipython
      tensorflow
      ipdb
      gdb
      ctags
      docker
    ] ++ tvmDeps;

    inherit tvmCmakeFlags;

    shellHook = ''
      if test -f /etc/myprofile ; then
        . /etc/myprofile
      fi

      mkdir .ipython-profile 2>/dev/null || true
      cat >.ipython-profile/ipython_config.py <<EOF
      print("Enabling autoreload")
      c = get_config()
      c.InteractiveShellApp.exec_lines = []
      c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
      c.InteractiveShellApp.exec_lines.append('%autoreload 2')
      EOF

      alias ipython='ipython --matplotlib=qt5 --profile-dir=.ipython-profile'

      TVM=tvm
      export PYTHONPATH="$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"

      cdtvm() { cd $TVM ; }
      cdex() { cd $TVM/nnvm/examples; }

      build() {(
        cdtvm ; sh ./tests/ci_build/ci_build.sh cpu ./tests/scripts/task_build.sh build -j5 ;
      )}

      test() {(
        cdtvm ; sh ./tests/ci_build/ci_build.sh cpu ./tests/scripts/task_python_nnvm.sh
      )}

      cdtvm
    '';
  };

}

