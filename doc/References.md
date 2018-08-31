Useful articles
===============

ML
--

### General

* https://arxiv.org/pdf/1409.2329.pdf
  2015, Zaremba, Recurrent Neural Network Regularization

### Examples and tutorials

* https://github.com/chiphuyen/stanford-tensorflow-tutorials
* https://github.com/aymericdamien/TensorFlow-Examples/
* https://github.com/adventuresinML/adventures-in-ml-code
* https://github.com/philipperemy/tensorflow-multi-dimensional-lstm

### General NLP

General articles:

* https://arxiv.org/pdf/1708.02709.pdf
  Recent Trends in Deep Learning Based Natural Language Processing
* http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
  LSTM tutorial

Occurences in Huawei mdeia:

* https://pdfs.semanticscholar.org/presentation/51d9/81c1b28818fd0ee94dd3e607e1004874dfef.pdf
  2015, Research on Deep Learning for Natural Language Processing at Huawei Noah’s Ark Lab
* https://www.huawei.com/en/about-huawei/publications/winwin-magazine/AI/intelligent-agents-tomorrow-digital-valets
  2016, Nuawei NLP news
* http://www.aclweb.org/anthology/N16-4004
  Noah ark document

### OCR

* https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791
  1998, LeCun, Gradient-Based Learning Applied to Document Recognition
* https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6628705
  2013, High-Performance OCR for Printed English and Fraktur using LSTM Networks
  - https://sourceforge.net/projects/rnnl/
    RNNLib - OpenSource library which was used by the authors.
* https://arxiv.org/pdf/1508.02774.pdf
  2015, Bruel, Benchmarking of LSTM Networks
* https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8
  2017, some Optical Character Recognition state-of-the-art article.
* https://arxiv.org/pdf/1805.09441.pdf
  2018, Implicit Language Model in LSTM for OCR


### ASR

Media

* https://www.slideshare.net/ssusercd5833/sequence-learning-with-ctc-technique
  Chun Hao Wang, Slides with lots of formuals

Articles

* https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf
  2014, Abdel-Hamid, Convolutional Neural Networks for Speech Recognition
* http://proceedings.mlr.press/v32/graves14.pdf
  2014, Graves, Towards End-to-End Speech Recognition with Recurrent Neural Networks
* http://homepages.inf.ed.ac.uk/llu/pdf/llu_icassp16.pdf
  2016, Lu, On training the recurrent neural network encoder-decoder for large vocabulary end-to-end speech recognition
* https://arxiv.org/pdf/1610.09975.pdf
  2016, Soltau, Neural Speech Recognizer: Acoustic-to-Word LSTM Model for Large Vocabulary Speech Recognition
* https://arxiv.org/pdf/1705.10874.pdf
  2017, Zhang, Deep Learning for Environmentally Robust Speech Recognition: An Overview of Recent Developments


### AD

Media

* http://vertex.ai/blog/fully-automatic-differentiation
* https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/
  2013, Introduction to Automatic Differentiation
* http://www.columbia.edu/~ahd2125/post/2015/12/5/
  2015, Automatic Differentiation or Mathemagically Finding Derivatives (blog, 2015). !Errors
* http://www.autodiff.org/?module=Introduction&submenu=Selected%20Books
  Collection of textbooks on AD

Articles

* http://www.bcl.hamilton.ie/~qobi/stalingrad/
  2005, Reverse-Mode AD in a Functional Framework: Lambda the Ultimate Backpropagator
* http://conal.net/papers/beautiful-differentiation/
  Forward-mode AD in Haskell, vector spaces
* https://arxiv.org/pdf/1711.01348
  2017, Automatic differentiation for tensor algebras, tech.report
* https://arxiv.org/pdf/1806.02136.pdf
  2018, Efficient Differentiable Programming in a Functional Array-Processing Language
* https://people.csail.mit.edu/tzumao/gradient_halide/gradient_halide.pdf
  2018, Differentiable Programming for Image Processing and Deep Learning in Halide


TVM
---

### General

* https://arxiv.org/pdf/1805.08166.pdf
  Learning to Optimize Tensor Programs

* https://arxiv.org/pdf/1802.04799.pdf
  TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

* https://github.com/dmlc/dmlc.github.io/blob/master/\_posts/2016-09-29-build-your-own-tensorflow-with-nnvm-and-torch.markdown
  How about build your own TensorFlow with NNVM and Torch7

* https://github.com/andersy005/tvm-in-action

* https://dl.acm.org/ft_gateway.cfm?id=3211348&ftid=1979148&dwn=1&CFID=8566367&CFTOKEN=e89fe1de14e82d30-9B3A947F-BA70-A2B8-C972CEE81188C1C5
  2018, Relay: A New IR for Machine Learning Frameworks

### Competitors

* https://www.tensorflow.org/performance/xla/
  XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear
  algebra that optimizes TensorFlow computations.
  - https://haosdent.gitbooks.io/tensorflow-document/content/resources/xla_prerelease.html

* https://github.com/facebookresearch/TensorComprehensions
  A domain specific language to express machine learning workloads.
  - https://arxiv.org/abs/1802.04730
    Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions

* https://github.com/plaidml/plaidml
  PlaidML - PlaidML is the easiest, fastest way to learn and deploy deep
  learning on any device, especially those running macOS or Windows.

* http://dlvm.org/
  - https://arxiv.org/pdf/1711.03016
    DLVM: A modern compiler infrastructure for deep learning systems

* https://github.com/vgvassilev/clad
  - https://llvm.org/devmtg/2013-11/slides/Vassilev-Poster.pdf
    clad - Automatic Differentiation using Clang

### Benchmarks

* https://github.com/dmlc/tvm/wiki/Benchmark
  TVM Benchmarking WIKI (remote devices for now)

* http://vertex.ai/blog/compiler-comparison
  [By PlaidML] Comparision between PlaidML, TVM, TensorComprehensions

* https://github.com/plaidml/plaidbench/tree/tensorcomp
  [By PlaidML] Benchmarks for Keras kernels, compares TVM and TC

* https://github.com/u39kun/deep-learning-benchmark

* https://knowm.org/deep-learning-frameworks-hands-on-review/
  General ML frameworks Review

### Related

* https://arxiv.org/pdf/1805.00907.pdf
  Glow: Graph Lowering Compiler Techniques for Neural Networks

* https://en.wikipedia.org/wiki/Polytope\_model

