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

#### 26.10.2018
* Instrument the whole [model](./model0v2.py) with checkpoints and shape
  pickers.
* Dumped the shapes of all the convolutions. There are 6 different shapes:
  (1,108,21,32)x3, (1,108,21,64)x35, (1,108,11,128)x35, (1,54,6,192)x20,
  (1,54,6,256)x20, (1,54,1,1318)x1
* TODO: Write a test comparing the performance of TVM vs TF on all of this
  shapes
* Compared the tolerance of TVM model with its TF equivalent. The
  precision is relatively low: 1e-1. Expected value is 1e-5.

#### 25.10.2018
* Compared the cumulative performance of block2 in TVM vs TF. There is no any
  operation which brings the speed down. Looks like every operation in block2 works
  a bit slower in TVM than in TF.
* ~~TODO: apply latest instrumentation approach to the whole model~~
* ~~TODO: compare shapes of the operations. Try to find 'cursed shapes'~~

#### 24.10.2018
* Obtained results from built-in profiler, which is called 'debugger' for some
  reason. The typical result of a single run is in
  [the dump file](./data/block2-timings-sorted.txt)
* We can see, that despite the fact that the slowest operation is 2-3 times
  slower than others, it can't cause x2 slowdown, so we need better
  comparision with TensorFlow
* Modified TVM block2 code to stop at specified place. TF already has this
  possibility from the box.

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
* Tested TF/TVM performance of repeated block1: TVM: 0.262+-0.034; TF:
  0.529+-0.007; Block1 was repeated 200 times. In both cases TVM is a clear
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
* Finished writing staging code, obtain Model sources in NNVM DSL, did some
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

