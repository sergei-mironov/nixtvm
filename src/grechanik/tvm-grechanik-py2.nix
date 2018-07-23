{ pkgs ?  import ../../nixpkgs {}
, stdenv ? pkgs.stdenv
} :

let
  inherit (pkgs) writeText fetchgit fetchgitLocal;
  inherit (builtins) filterSource;
  inherit (pkgs.lib.sources) cleanSourceFilter;

  pp = pkgs.python27Packages;
in

rec {


  shell = stdenv.mkDerivation {
    name = "shell";

    buildInputs = (with pkgs; [
      python2
      cmake
      ncurses
      zlib
      gdb
      universal-ctags
      docker
      llvm_5
      clang
      gtest
      openblas
    ]) ++ (with pp; [
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
      h5py
      pip
      pillow
      pylint
    ]);

    shellHook = ''
      #if test -f /etc/myprofile ; then
      #  . /etc/myprofile
      #fi

      #if test -f ~/.display ; then
      #  . ~/.display
      #fi

      # Fix g++(v7.3): error: unrecognized command line option ‘-stdlib=libstdc++’; did you mean ‘-static-libstdc++’?
      unset NIX_CXXSTDLIB_LINK

      export PYTHONPATH="$PYTHONPATH:$HOME/.local/lib/python3.6/site-packages" 
    '';
  };

}

