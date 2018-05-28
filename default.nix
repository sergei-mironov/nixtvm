{ pkgs ? import <nixpkgs> {}
, stdenv ? pkgs.stdenv
}:
stdenv.mkDerivation {
  name = "buildenv";
  buildInputs = [
    ../nnvm
  ];

  shellHook = ''
    if test -f /etc/myprofile ; then
      . /etc/myprofile
    fi
  '';
}


