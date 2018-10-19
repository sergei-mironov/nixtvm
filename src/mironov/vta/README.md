TVM VTA
=======

Versatile Tensor Accelerator (VTA) is a FPGA-based backed of TVM. It allows to
offload the resource-intensive computations from CPU to the RISC co-processor
supporting whole-tensor instructions. According to the current approach, this
co-processor is implemented as an IP-circuit of FPGA board.


Links
=====

* https://tvm.ai/vta
  Main VTA page

* https://discuss.tvm.ai/search?q=VTA
  TVM forum filtered by the VTA keyword

* https://www.xilinx.com/support/university/boards-portfolio/xup-boards.html
  List of low-cost Xilinx boards. PYNQ boards are reported to be compatible with
  VTA

* https://www.xilinx.com/products/design-tools/vivado.html#buy
  Vivado Design Suite pricing. Approx. $4000 per license. Probably, it is not
  required to run the VTA.

* https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html
  Vivado Higl-Level Synthesis (Vivado HLS) used by TVM. HLS is a part of
  Vivado suite.


Log
===

#### 19.10.2018
* Asked question on TVM forum. [Link](https://discuss.tvm.ai/t/support-for-pynq-v2-3-image/940/4)

#### 18.10.2018
* Received a question from https://linux.org.ru forum regarding VTA.
* Collect links to determine the price of equpment required to join the VTA
  development. Result are:
  - $4000 Xilinx Vivado IDE
  - $200 PYNQ board

