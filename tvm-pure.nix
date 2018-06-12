{ pkgs ?  import <nixpkgs> {}
, stdenv ? pkgs.stdenv
} :

import ./tvm.nix {
  tvmCmakeFlags = "";
  tvmDeps = [];
}

