Components of recurrent networks
================================

 * Simple Recurrent Neural Networks (RNN)
 * Bidirectional RNN
 * Encoder-Decoder Architecures
 * Gated Recurrent Unit (GRU)
 * Long Short-term Memory (LSTM)
 * Memory Networks
 * Neural Turing Machine (NTM)
 * Attention Mechanism
 * Dynamic Recurrent Neural Network (DRNN)


NLP large-scale tasks
=====================

1. Applying different types of word embedding in TVM:
  * Models which use CBOW-based word embedding
  * Models which use Word2Vec-based methods
  * Inference of models which use character-embedding methods (suitable for Chinese)

2. Applying different types of Convolutional NNs for solving NLP tasks using TVM
  *	Basic CNN (TVM needs lookup-tables primitive)
  *	Window-based approach
  *	Time-delay neural networks (TDNN)
  *	Dynamic Convolutional NN (DCNN)
  *	Dynamic multy-pooling convolutional NN

3. Applying different types of Recurrent networks for solving NLP tasks using
   TVM (handled by Amit’s team up to some degree)
  * Simple Recurrent NN (RNN)
  * Long-short term memory LSTM
  * Gated recurrent units
  * Bi-directional Recurrent Neural Network
  * Dynamic Recurrent Neural Network

4. Applying TVM to handle attention-based neural networks (interesting task!)
  * Combining CNN and RNN to simulate attention behavior

5. Applying TVM to handle Recursive Neural Networks
  * Running MV-RNN model with TVM
  * Running RNTN model with TVM

6. Investigate the possibilities of using TVM for running Deep Reinforced models
   and Deep Unsupervised learning

7. Investigate the possibilities of using TVM for running Deep Generative Models
  * Start from running simpler GAN models using TVM
  * Continue by combining GANs with LSTM and attention mechanism

