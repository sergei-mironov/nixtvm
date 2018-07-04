TODO
====

Loose list:

 1. Brief story of [Halide](http://halide-lang.org/), some figures from Halide
    article/video.
    <br/>
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
    <br/>
 10. TVM scheduling algorithms
 20. TVM benchmarks, able to prove the results
      * https://www.sysml.cc/doc/78.pdf
      * https://dl.acm.org/citation.cfm?doid=3229762.3229764
 30. NNVM story. Role of NNVM in the TVM/NNVM infrastructure
 40. NNVM compiler pipeline

