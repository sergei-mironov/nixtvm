TODO
====

* ~~Run `run` several times?~~
* ~~Try with default NNVM optimisations disabled~~
* Measure all possible times (Wall time, CPU time, etc)
* Figure out parallelism. Which module does schedule parallel execution in TVM?
* Try with batch size ~100?
* Compare TF/TVM performance on different segments of the Model.

Problems
========

* Simple TF runners use `session.run` which may be slow, try feeders.
* TV/TVM error correlates with absolute input value, setup relative tolerance.


LOG
===

#### 23.10.2018
* Finished block2 in TVM. Results of running a single instance of block2 are:
  - NNVM: 0.047517
  - TF:   0.026365
  We see that TVM is twice as fast as TVM on this block
  TVM/NNVM was compiled with debug runtime
* Noticed the autotvm missing workload warning issued by TVM
* More experiments showed even worse time for TVM/NNVM: 0.06411564

#### 18.10.2018
* Block2, fix bugs in TVM DSL for block2. `block2_block_nnvm` now compiles.

#### 16.10.2018
* Implemented TF runner for block2. Turns out that it is possible to run the
  part of the model in TensorFlow. TF checks that users provide enough
  information to the Model to run it.

#### 15.10.2018
* Started block2 implementation

#### 04.10.2018
* Tested TF/TVM performance of repeted block1: TVM: 0.262+-0.034; TF:
  0.529+-0.007; Block1 was repeated 200 times in both cases TVM is a clear
  winner.
* Fixed error in TF code of `block1` (sqrt`->`rsqrt). Now results are equal
  with absoulte tolerance of 1e-5.
* Decided to compare TF/TVM performance for model block between
  `Rcnn_ctcV3/expand_conv3/add_13/add` and `Rcnn_ctcV3/expand_conv4/add_17/add`
  nodes.

#### 01.10.2018
* Manually encoding TF-version of block between `Rcnn_ctcV3/expand_conv1/add_1`
  and `Rcnn_ctcV3/conv_block1/unit1/add_2/add` nodes. Its name is `block`.
* Strange difference in results between TF and TVM version of block1.

#### 27.09.2018
* Finished writing staging code, obtain Model sources in NNVM DSL, do some
  tests.
* Plan `partsearch` experiment which would measure the performance of the Model
  running from the same input node to different output nodes.

![partsearch](./partsearch.png)

* Main experiment results are on image above.
  Points on Y-axis are execution times in seconds.
  Points on X-axis correspond to the folowing nodes:

  - 0: `Rcnn_ctcV3/expand_conv1/add_1/add`
  - 1: `Rcnn_ctcV3/expand_conv2/add_7/add`
  - 2: `Rcnn_ctcV3/expand_conv3/add_13/add`
  - 3: `Rcnn_ctcV3/expand_conv4/add_17/add`
  - 4: `Rcnn_ctcV3/conv2d_116/BiasAdd`

  One can see that TVM is faster on smaller parts of the model, but performance
  seems to degrade on larger parts.
* Decided to measure the performance of specific parts of the Model taking
  different input nodes. We will call them `blocks`.


#### 13.09.2018
* Measured the performance of the Model using TVM and Tensorflow.
  TF shows better results.
* Decided to measure the performance on a specific model parts. Unfortunately,
  TVM doesn't provide access to intermediated blocks of imported models.
* Started the implementation of staging module able to produce the NNVM DSL code
  during TF importing

#### 01.09.2018
* Obtained `RCNN_ctcV3` model

