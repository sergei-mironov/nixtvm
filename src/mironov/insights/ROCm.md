ROCm
====

Current ROCm version: 1.9.1

Sites:

* https://rocm.github.io/
  ROCm website

* https://rocm-documentation.readthedocs.io/en/latest/index.html
  ROCm docs

* https://www.khronos.org/conformance/adopters/conformant-products
  Vulkan conformance (not related to ROCm)

Videos:

* https://www.youtube.com/watch?v=k3aGaxcYCxw
  2016, Radeon Open Compute (ROCm) Platform Discussion
  - An AMD-based driver interface, related to Radeon.
  - Bring ML deeper and deepre to core

* https://www.youtube.com/watch?v=U6EuSiR8ooM
  2018, AMD ROCm 2.0 Session 2 Nov 6 2018 Next Horizon
  Slides shown:
  - ROCm based applications
  - Option1, for languages : LLVM -> AMDGPU Compiler
  - Option2, Graph compilers:
    * TVM(!) -> AMDGPU Compiler
    * XLA -> AMDGPU, Note: XLA should support ROCM, as ROCm guys say
  - Libraries for ML and HPC(?) applications: rocBLAS, rocSparse, rocRAND, etc.
  - Multy-gpu communications: RCCL, RDMA, ROCm enabled UCX
  - MIopen (?)
  - Performance slide, ResNet50
  - Frameworks: Tf, PyTorch, Caffe2
  - Deploy: Docker, Kubernetes
  - Cloud deploy: GPUeater
  - More development in 2019
