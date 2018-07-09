{ pkgs ?  import ./nixpkgs {}
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


  mktags = pkgs.writeShellScriptBin "mktags" ''
    (
    cd $CWD
    find tvm -name '*cc' -or -name '*hpp' -or -name '*h' | \
      ctags -L - --c++-kinds=+p --fields=+iaS --extra=+q --language-force=C++
    find tvm src _tags -name '*py' | \
      ctags --append -L -
    cat tags | grep -v -w import | grep -v -w _io_ops | grep -v -w 'ops\.' > tags.2
    mv tags.2 tags
    )
  '';


  shell = stdenv.mkDerivation {
    name = "shell";

    buildInputs = (with pkgs; [
      cmake
      ncurses
      zlib
      mktags
      gdb
      universal-ctags
      docker
      llvm
      clang
      gtest
    ]) ++ (with pp; [
      tensorflow
      decorator
      tornado
      ipdb
      nose
      pyqt5
      numpy
      scikitlearn
      matplotlib
      ipython
      jupyter
      scipy
      mxnet_localssl
      onnx
    ]) ++ tvmDeps;

    inherit tvmCmakeFlags;

    shellHook = ''
      if test -f /etc/myprofile ; then
        . /etc/myprofile
      fi

      if test -f ~/.display ; then
        . ~/.display
      fi

      export CWD=`pwd`
      mkdir .ipython-profile 2>/dev/null || true
      cat >.ipython-profile/ipython_config.py <<EOF
      print("Enabling autoreload")
      c = get_config()
      c.InteractiveShellApp.exec_lines = []
      c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
      c.InteractiveShellApp.exec_lines.append('%autoreload 2')
      EOF

      if test -n "$DISPLAY"; then
        export QT_QPA_PLATFORM_PLUGIN_PATH=`echo ${pkgs.qt5.qtbase.bin}/lib/qt-*/plugins/platforms/`
        alias ipython='ipython --matplotlib=qt5 --profile-dir=$CWD/.ipython-profile'
        alias ipython0='ipython --profile-dir=$CWD/.ipython-profile'
      fi

      export TVM=$CWD/tvm
      export PYTHONPATH="$CWD/src/$USER:$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"
      export LD_LIBRARY_PATH="$TVM/build-native:$LD_LIBRARY_PATH"
      export C_INCLUDE_PATH="$TVM/include:$TVM/dmlc-core/include:$TVM/HalideIR/src:$TVM/dlpack/include:$TVM/topi/include:$TVM/nnvm/include"
      export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
      export LIBRARY_PATH=$TVM/build-native

      # Fix g++(v7.3): error: unrecognized command line option ‘-stdlib=libstdc++’; did you mean ‘-static-libstdc++’?
      unset NIX_CXXSTDLIB_LINK

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

      cdc() {(
        cd $CWD
      )}
    '';
  };

}

