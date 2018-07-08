TODO
====

 1. Brief story of [Halide](http://halide-lang.org/), some figures from Halide
    article/video.
     * [Halide video](https://youtu.be/3uiEyEKji0M)
    Halide achivements:
     * FCam raw pipeline (Denoise -> Demosaic -> Color correct -> Tone curve)
       5% faster, 2.75x less code
     * Local Laplacian filters (Permaid-based, increased local contrast)
       Baseline used OpenMP and Intel Performance Primitives (IPP)
       2x faster (CPU), 7x faster (GPU), 4x less code
     * Bilateral Grid (Grid -> Bluring -> Slicing)
       5.9x faster (CPU), 2x faster (GPU), 3x less code
     * Snake image segmentation (Li et all, 2010)
       70x faster (CPU), 1250x faster (GPU), 2x bigger (matlab -> C++)
 1. TVM Open Source Project
     * Mostly made by a single person (Tianqi Chen + tqchen ~= 50%)
 1. TVM basics
     * Model sources (MXnet, ONNX, TF(recently added), Keras, Darknet)
     * Build targets ( ... FPGA (Verilog), Web(!))
 1. TVM scheduling algorithms (based on `lection7.pdf`)
     * (CPU,GPU) Loop Transformations
        * Split
        * Recorder ???
     * (CPU, GPU) Thread bindings
        * Thread axis  (threadIdx, blockIdx)
     * (CPU) Cache locality
     * (GPU) Thread cooperation
     * (GPU) Tensorisation
     * (GPU) Latency hiding
 1. TVM fusion
     * TODO: Find out info about fusion algorithms
 1. NNVM basics
 1. NNVM compiler pipeline
 1. NNVM training / autograd status ???
 1. Competitors
     * TensorComprehensions (Genetic algorithms)
     * PlaidML (non-NVidia orientation, Tile language)
     * DLVM (IR-backed differentiation)
     * clad
 1. Benchmarks available. Find out how to reproduce the results presented in
    the following articles:
     * https://www.sysml.cc/doc/78.pdf
     * https://dl.acm.org/citation.cfm?doid=3229762.3229764
 1. Development plans

