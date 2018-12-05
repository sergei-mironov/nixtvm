# For CPU
FROM ubuntu:16.04

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh

COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

COPY install/proxy.sh /install/proxy.sh
RUN bash /install/proxy.sh

COPY install/ubuntu_install_llvm.sh /install/ubuntu_install_llvm.sh
RUN bash /install/ubuntu_install_llvm.sh

COPY install/ubuntu_install_golang.sh /install/golang.sh
RUN bash /install/golang.sh

RUN pip install cpplint
RUN apt-get install -y gdb
RUN apt-get install -y apport

# Runs into https://bugs.launchpad.net/ubuntu/+source/python3.6/+bug/1631367 otherwise
RUN rm /usr/lib/python3.6/sitecustomize.py
