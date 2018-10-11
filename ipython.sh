#!/bin/sh

if ! test -d "$CWD" ; then
  echo "CWD is not set"
  exit 1
fi

# if test -n "$DISPLAY"; then
#   export QT_QPA_PLATFORM_PLUGIN_PATH=`echo ${pkgs.qt5.qtbase.bin}/lib/qt-*/plugins/platforms/`
#   alias ipython='ipython --matplotlib=qt5 --profile-dir=$CWD/.ipython-profile'
#   alias ipython0='ipython --profile-dir=$CWD/.ipython-profile'
# fi

mkdir .ipython-profile 2>/dev/null || true
cat >.ipython-profile/ipython_config.py <<EOF
c = get_config()
c.InteractiveShellApp.exec_lines = []
c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
c.InteractiveShellApp.exec_lines.append('%autoreload 2')

def tweak():
  print("Enabling tweaks")

  import numpy as np
  np.set_printoptions(edgeitems=30, linewidth=100000)

  import ssl;
  ssl._create_default_https_context = ssl._create_unverified_context

  import matplotlib;
  matplotlib.use('agg');

  import matplotlib.pyplot;
  matplotlib.pyplot.ioff()

tweak()
EOF

ipython3 --profile-dir=$CWD/.ipython-profile -i "$@"


