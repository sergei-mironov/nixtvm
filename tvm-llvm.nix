{ pkgs ?  import <nixpkgs> {}
, stdenv ? pkgs.stdenv
} :

import ./tvm.nix {
  tvmCmakeFlagsEx = "-DUSE_LLVM=ON";
  tvmDepsEx = with pkgs; [ llvm_5 ];
}

