TODO
====

Loose list:

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
 05. TVM basics
 10. TVM scheduling algorithms (based on `lection7.pdf`)
     * (CPU,GPU) Loop Transformations
        * Split
        * Recorder ???
     * (CPU, GPU) Thread bindings
        * Thread axis  (threadIdx, blockIdx)
     * (CPU) Cache locality
     * (GPU) Thread cooperation
     * (GPU) Tensorisation
     * (GPU) Latency hiding
 15. TVM fusion
     TODO: Find out info about fusion algorithms
 30. NNVM basics
 40. NNVM compiler pipeline
 45. NNVM training / autograd status
 50. Benchmarks available. Find out how to reproduce the results presented in
     the following articles:
     * https://www.sysml.cc/doc/78.pdf
     * https://dl.acm.org/citation.cfm?doid=3229762.3229764

