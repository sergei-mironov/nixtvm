{ pkgs ?  import <nixpkgs> {}
, stdenv ? pkgs.stdenv
, tvmCmakeFlagsEx ? abort "Use tvm-<mode>.nix"
, tvmDepsEx ? abort "Use tvm-<mode>.nix"
, tvmCmakeConfig ? ""
} :

let
  inherit (pkgs) writeText fetchgit fetchgitLocal;
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


  mktags = pkgs.writeShellScriptBin "mktags" ''
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
      pyqt5
      llvm
      clang
    ] ++ tvmDeps;

    inherit tvmCmakeFlags;

    shellHook = ''
      if test -f /etc/myprofile ; then
        . /etc/myprofile
      fi

      CWD=`pwd`
      mkdir .ipython-profile 2>/dev/null || true
      cat >.ipython-profile/ipython_config.py <<EOF
      print("Enabling autoreload")
      c = get_config()
      c.InteractiveShellApp.exec_lines = []
      c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
      c.InteractiveShellApp.exec_lines.append('%autoreload 2')
      EOF
      export QT_QPA_PLATFORM_PLUGIN_PATH=`echo ${pkgs.qt5.qtbase.bin}/lib/qt-*/plugins/platforms/`

      alias ipython='ipython --matplotlib=qt5 --profile-dir=$CWD/.ipython-profile'
      alias ipython0='ipython --profile-dir=$CWD/.ipython-profile'

      export TVM=$CWD/tvm
      export PYTHONPATH="$CWD/src/tutorials:$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"
      export LD_LIBRARY_PATH="$TVM/build-native:$LD_LIBRARY_PATH"
      export C_INCLUDE_PATH="$TVM/include:$TVM/dmlc-core/include:$TVM/HalideIR/src:$TVM/dlpack/include:$TVM/topi/include:$TVM/nnvm/include"
      export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
      export LIBRARY_PATH=$TVM/build-native

      cdtvm() { cd $TVM ; }
      cdex() { cd $TVM/nnvm/examples; }

      nmake() {(
        cdtvm
        mkdir build-native 2>/dev/null
        cat ${writeText "cfg" tvmCmakeConfig} >build-native/config.cmake
        cd build-native
        cmake ..
        make -j6 "$@"
      )}

      nclean() {(
        cdtvm
        cd build-native
        make clean
        rm CMakeCache.txt
        rm -rf CMakeFiles
      )}

      dmake() {(
        cdtvm
        mkdir build 2>/dev/null
        cat ${writeText "cfg" tvmCmakeConfig} >build/config.cmake
        sh ./tests/ci_build/ci_build.sh cpu ./tests/scripts/task_build.sh build -j6 "$@";
      )}

      alias build="dmake"

      dtest() {(
        cdtvm
        sh ./tests/ci_build/ci_build.sh cpu ./tests/scripts/task_python_nnvm.sh
      )}

      cdtvm
    '';
  };

}

