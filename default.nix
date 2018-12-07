{ pkgs ?  import ./nixpkgs {}
, stdenv ? pkgs.stdenv
} :

let
  inherit (pkgs) writeText fetchgit fetchgitLocal;
  inherit (builtins) filterSource;
  inherit (pkgs.lib.sources) cleanSourceFilter;

  pypkgs = pkgs.python36Packages;
in

rec {


  mktags = pkgs.writeShellScriptBin "mktags" ''
    (
    cd $CWD
    find src tvm -name '*cc' -or -name '*hpp' -or -name '*h' -or -name '*\.c' -or -name '*cpp' | \
      ctags -L - --excmd=number --c++-kinds=+p --fields=+iaS --extras=+q --language-force=C++

    while test -n "$1" ; do
      case "$1" in
        py)
          find tvm src -name '*py' | ctags --excmd=number --append -L -
          ;;
        tf)
          echo "Building Tensorflow tags" >&2
          find -L _tags/tensorflow -name '*py' | ctags --append -L -
          cat tags | grep -v -w import | grep -v -w _io_ops | grep -v -w 'ops\.' > tags.2
          mv tags.2 tags
          find -L _tags/tensorflow -name '*cc' -or -name '*h' | ctags --append --language-force=C++ -L -
          ;;
        *)
          echo "Unknown tag task: $1" >&2
          ;;
      esac
      shift
    done
    )
  '';


  tvm-env = stdenv.mkDerivation {
    name = "tvm-env";

    buildInputs = (with pkgs; [
      cmake
      ncurses
      zlib
      mktags
      gdb
      universal-ctags
      docker
      gtest
      llvm_6
      clang_6
      openblas
      tig
    ]) ++ (with pypkgs; [
      Keras
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
      # mxnet_localssl
      # onnx
      pillow
      pylint
    ]);

    shellHook = ''
      if test -f /etc/myprofile ; then
        . /etc/myprofile
      fi

      if test -f ~/.display ; then
        . ~/.display
      fi

      # See https://github.com/NixOS/nixpkgs/issues/38991
      export LOCALE_ARCHIVE=${pkgs.glibcLocales}/lib/locale/locale-archive

      export CWD=`pwd`

      export TVM=$CWD/src/$USER/tvm
      export BUILD=build-native
      export PYTHONPATH="$CWD/src/$USER:$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"
      export LD_LIBRARY_PATH="$TVM/$BUILD:$LD_LIBRARY_PATH"
      export C_INCLUDE_PATH="$TVM/include:$TVM/3rdparty/dmlc-core/include:$TVM/3rdparty/HalideIR/src:$TVM/3rdparty/dlpack/include:$TVM/topi/include:$TVM/nnvm/include"
      export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
      export LIBRARY_PATH=$TVM/$BUILD

      # Fix g++(v7.3): error: unrecognized command line option ‘-stdlib=libstdc++’; did you mean ‘-static-libstdc++’?
      unset NIX_CXXSTDLIB_LINK NIX_TARGET_CXXSTDLIB_LINK

      cdc() { cd $CWD ;  }
      cdtvm() { cd $TVM ; }

      alias ipython="$CWD/ipython.sh"

      nmake() {(
        cdtvm
        mkdir "$BUILD" 2>/dev/null
        cp $TVM/cmake/config.cmake $TVM/$BUILD/config.cmake
        sed -i 's/USE_LLVM OFF/USE_LLVM ON/g' $TVM/$BUILD/config.cmake
        (
          cd "$BUILD"
          cmake ..
          make "$@" -j6
        ) && ln -sfv --no-dereference "$BUILD" build # FIXME: Python uses 'build' name
      )}

      nclean() {(
        cdtvm
        cd $BUILD
        make clean
        rm CMakeCache.txt
        rm -rf CMakeFiles
      )}

      ntest() {(
        cdtvm
        sh ./tests/ci_build/ci_build.sh cpu ./tests/scripts/task_python_nnvm.sh
      )}

    '';
  };

}.tvm-env
