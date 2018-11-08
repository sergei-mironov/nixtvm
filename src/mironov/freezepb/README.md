TODO
====

* ~~Run `run` several times?~~
* ~~Try with default NNVM optimisations disabled~~
* ~~Compare TF/TVM performance on different segments of the Model.~~
* Measure all possible times (Wall time, CPU time, etc)
* Figure out parallelism. Which module does schedule parallel execution in TVM?
* Try with batch size ~100?
* Try to apply scheduling
* Try to enable autotuner

Problems
========

* Simple TF runners use `session.run` which may be slow, try feeders.
* TV/TVM error correlates with absolute input value, setup relative tolerance.
* `nnvm_shape` in ./convperf.py doesn't work for some reason

Vocabulary
==========

 * Model - the `RCN_ctcV3` model, available originally as TF Protobuf file
 * Block1 - head part of the model, approx 1/4 from the beginning
 * Block2 - tail part of the mode, approx 1/3 from the end


Slides
======

 * [Stauts report 2018-10-10](http://code.huawei.com/mrc-cbg-opensource/hitvm-internal/blob/master/mironov/doc/Meeting%202018-10-10/MRC%20OSI%20Status%20report%202018-10-10.pptx)


Folder description
==================

    .
    ├── modeldefs.py       - Define global model constants and helpers
    ├── runners.py         - Define `with_nnvm`, `with_tf` and other helper combinators
    ├── model0v2.py        - Latest version of Model definition in NNVM
    ├── convperf.py        - Experiments on individual convolution operations
    ├── block1.py          - ** May be out of date** Encodes experiments on Block1 of the Model
    ├── block2.py          - ** May be out of date** Encodes experiments on Block2 of the Model
    ├── block2v2.py        - ** May be out of date** Encodes experiments on Block2 of the Model
    ├── data
    │   └── block2-timings-sorted.txt
    ├── main.py            - ** Out of date ** First experiments on the model
    ├── model0.py          - ** Out of date ** First version of model0v2.py
    ├── partsearch.json
    ├── partsearch.png
    └── README.md          - This README

LOG
===

#### 01.11.2018
 * Learned about scheduling in NNVM.

#### 30.10.2018
 * Below are convolution execution results, sorted by model time.  Times are
   in seconds, every type of convolution block was repeated 200 times for
   reliable measurements

       (1, 54, 6, 192)   kernel (3, 3)  time1 0.485376 timeM 3.88301
       (1, 108, 11, 128) kernel (3, 3)  time1 0.263262 timeM 3.15915
       (1, 54, 6, 192)   kernel (1, 1)  time1 0.119617 timeM 1.79426
       (1, 54, 6, 256)   kernel (3, 3)  time1 0.247189 timeM 1.48313
       (1, 54, 1, 1318)  kernel (1, 1)  time1 1.146242 timeM 1.14624
       (1, 108, 21, 64)  kernel (3, 3)  time1 0.091557 timeM 1.09869
       (1, 108, 11, 128) kernel (1, 1)  time1 0.039408 timeM 0.90638
       (1, 54, 6, 256)   kernel (1, 1)  time1 0.042709 timeM 0.55522
       (1, 108, 21, 64)  kernel (1, 1)  time1 0.014288 timeM 0.32863

   We see that we should optimize (1,54,6,192) and (1,108,11,128) kernels
   first in this model since we spend most of the time there..

#### 29.10.2018
 * Wrote simple conv2d testbench for model's typical shapes. (3x3) kernels work
   x10 times slower than trivial 1x1 kernels. See [test sources](./convperf.py)
 * TODO: interpret the results

#### 26.10.2018
 * Instrumented the whole [model](./model0v2.py) with checkpoints and shape
   pickers.
 * Dumped the shapes of all the convolutions. There are 6 different shapes:
   - (1,108,21,32)x3    kernel 11x31
   - (1,108,21,64)x35   kernels 1x1 3x3
   - (1,108,11,128)x35  kernels 1x1 3x3
   - (1,54,6,192)x20    kernels 1x1 3x3
   - (1,54,6,256)x20    kernels 1x1 3x3
   - (1,54,1,1318)x1    kernel 1x1
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

