import tensorflow as tf

from nnvm import sym as _sym

from tensorflow import Tensor as TF_Tensor
from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph as TF_Graph, GraphDef as TF_GraphDef
from tensorflow.python.ops import variables

from freezepb.runners import *
from freezepb.modeldefs import *

    # sym_382705136 = tf.nn.conv2d( sym_457698256,sym_75044384,
    #     padding="VALID",data_format="NHWC",strides=(1,1,1,1),name="Rcnn_ctcV3/expand_conv1/conv2d_4/convolution")


def model_const_tf(name,shape):
  assert tuple(MODEL_PARAMS[name].shape)==tuple(shape)
  return tf.constant(MODEL_PARAMS[name].asnumpy())

def model1_block_consts():
  return [
    "Rcnn_ctcV3/expand_conv1/conv2d_4/kernel",
    "Rcnn_ctcV3/expand_conv1/conv2d_4/bias",
    "Rcnn_ctcV3/expand_conv1/activation/conv2d_6/kernel",
    "Rcnn_ctcV3/expand_conv1/activation/conv2d_6/bias",
    "Rcnn_ctcV3/expand_conv1/conv2d_5/kernel",
    "Rcnn_ctcV3/expand_conv1/conv2d_5/bias",
    "Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/gamma",
    "Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/beta",
    "Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/moving_mean",
    "Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/moving_variance",
    "Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/add/y",
    "Rcnn_ctcV3/expand_conv1/activation/conv2d_7/kernel",
    "Rcnn_ctcV3/expand_conv1/activation/conv2d_7/bias",
    "Rcnn_ctcV3/expand_conv1/activation/max_2/mul/x",
    "Rcnn_ctcV3/expand_conv1/conv2d_8/kernel",
    "Rcnn_ctcV3/expand_conv1/conv2d_8/bias"]


MODEL1_BLOCK_PARAMS={k:MODEL_PARAMS[k] for k in model1_block_consts()}

def model1_block_nnvm_consts():
  return {
  "sym_75044384"  : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/conv2d_4/kernel",shape=(1, 1, 32, 64)),
  "sym_73427696"  : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/conv2d_4/bias",shape=(64,)),
  "sym_223382672" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_6/kernel",shape=(1, 1, 64, 64)),
  "sym_356827536" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_6/bias",shape=(64,)),
  "sym_451228704" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/conv2d_5/kernel",shape=(3, 3, 32, 64)),
  "sym_88828560"  : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/conv2d_5/bias",shape=(64,)),
  "sym_379167584" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/gamma",shape=(64,)),
  "sym_492256464" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/beta",shape=(64,)),
  "sym_104779696" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/moving_mean",shape=(64,)),
  "sym_378983504" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/moving_variance",shape=(64,)),
  "sym_73418512"  : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/add/y",shape=(1,)),
  "sym_134609696" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_7/kernel",shape=(1, 1, 64, 64)),
  "sym_131967104" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_7/bias",shape=(64,)),
  "sym_473740336" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/activation/max_2/mul/x",shape=(1,)),
  "sym_112766576" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/conv2d_8/kernel",shape=(3, 3, 64, 64)),
  "sym_105635760" : _sym.Variable(name="Rcnn_ctcV3/expand_conv1/conv2d_8/bias",shape=(64,))
  }

def model1_block_nnvm(sym_149080784, sym_consts):
  # Begin of Cell 1
  cs=sym_consts
  sym_75044384  = cs["sym_75044384"]
  sym_73427696  = cs["sym_73427696"]
  sym_223382672 = cs["sym_223382672"]
  sym_356827536 = cs["sym_356827536"]
  sym_451228704 = cs["sym_451228704"]
  sym_88828560  = cs["sym_88828560"]
  sym_379167584 = cs["sym_379167584"]
  sym_492256464 = cs["sym_492256464"]
  sym_104779696 = cs["sym_104779696"]
  sym_378983504 = cs["sym_378983504"]
  sym_73418512  = cs["sym_73418512"]
  sym_134609696 = cs["sym_134609696"]
  sym_131967104 = cs["sym_131967104"]
  sym_473740336 = cs["sym_473740336"]
  sym_112766576 = cs["sym_112766576"]
  sym_105635760 = cs["sym_105635760"]

  sym_457698256 = _sym.pad(sym_149080784,pad_width=((0, 0), (0, 0), (0, 0), (0, 0)))
  sym_382705136 = _sym.conv2d(sym_457698256,sym_75044384,padding=[0, 0],dilation=(1, 1),layout="NHWC",strides=(1, 1),kernel_size=(1, 1),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/conv2d_4/convolution",use_bias=False)
  sym_457698256 = _sym.broadcast_add(sym_382705136,sym_73427696)
  sym_394053216 = _sym.pad(sym_149080784,pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))
  sym_140340480 = _sym.conv2d(sym_394053216,sym_451228704,dilation=(1, 1),layout="NHWC",strides=(1, 1),padding=[0, 0],kernel_size=(3, 3),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/conv2d_5/convolution",use_bias=False)
  sym_394053216 = _sym.broadcast_add(sym_140340480,sym_88828560)
  sym_88729488 = _sym.broadcast_add(sym_378983504,sym_73418512,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/add")
  sym_104808848 = _sym.__pow_scalar__(sym_88729488,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/Rsqrt",scalar=-0.5)
  sym_80975232 = _sym.broadcast_mul(sym_104808848,sym_379167584,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/mul")
  sym_86811088 = _sym.broadcast_mul(sym_394053216,sym_80975232,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/mul_1")
  sym_382126160 = _sym.broadcast_mul(sym_104779696,sym_80975232,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/mul_2")
  sym_104808912 = _sym.broadcast_sub(sym_492256464,sym_382126160,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/sub")
  sym_114622080 = _sym.broadcast_add(sym_86811088,sym_104808912,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/add_1")
  sym_382620512 = _sym.pad(sym_114622080,pad_width=((0, 0), (0, 0), (0, 0), (0, 0)))
  sym_457310224 = _sym.conv2d(sym_382620512,sym_223382672,layout="NHWC",strides=(1, 1),padding=[0, 0],dilation=(1, 1),kernel_size=(1, 1),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/activation/conv2d_6/convolution",use_bias=False)
  sym_382620512 = _sym.broadcast_add(sym_457310224,sym_356827536)
  sym_74589536 = _sym.pad(sym_114622080,pad_width=((0, 0), (0, 0), (0, 0), (0, 0)))
  sym_73915248 = _sym.conv2d(sym_74589536,sym_134609696,layout="NHWC",strides=(1, 1),padding=[0, 0],dilation=(1, 1),kernel_size=(1, 1),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/activation/conv2d_7/convolution",use_bias=False)
  sym_74589536 = _sym.broadcast_add(sym_73915248,sym_131967104)
  sym_73915280 = _sym.broadcast_add(sym_382620512,sym_74589536,name="Rcnn_ctcV3/expand_conv1/activation/max_2/add")
  sym_73550304 = _sym.broadcast_sub(sym_382620512,sym_74589536,name="Rcnn_ctcV3/expand_conv1/activation/max_2/sub")
  sym_457788432 = _sym.relu(sym_73550304,name="Rcnn_ctcV3/expand_conv1/activation/max_2/Relu")
  sym_457788880 = _sym.broadcast_add(sym_73915280,sym_457788432,name="Rcnn_ctcV3/expand_conv1/activation/max_2/add_1")
  sym_105411888 = _sym.broadcast_sub(sym_74589536,sym_382620512,name="Rcnn_ctcV3/expand_conv1/activation/max_2/sub_1")
  sym_76993232 = _sym.relu(sym_105411888,name="Rcnn_ctcV3/expand_conv1/activation/max_2/Relu_1")
  sym_223578592 = _sym.broadcast_add(sym_457788880,sym_76993232,name="Rcnn_ctcV3/expand_conv1/activation/max_2/add_2")
  sym_152477424 = _sym.broadcast_mul(sym_473740336,sym_223578592,name="Rcnn_ctcV3/expand_conv1/activation/max_2/mul")
  sym_393365024 = _sym.pad(sym_152477424,pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))
  sym_79064400 = _sym.conv2d(sym_393365024,sym_112766576,dilation=(1, 1),layout="NHWC",strides=(1, 1),padding=[0, 0],kernel_size=(3, 3),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/conv2d_8/convolution",use_bias=False)
  sym_393365024 = _sym.broadcast_add(sym_79064400,sym_105635760)
  sym_118484944 = _sym.broadcast_add(sym_393365024,sym_457698256,name="Rcnn_ctcV3/expand_conv1/add_1/add")
  # End of Cell 1
  return sym_118484944

def model1_block_tf(sym_149080784):
  # Begin of Cell 1
  sym_75044384  = model_const_tf(name="Rcnn_ctcV3/expand_conv1/conv2d_4/kernel",shape=(1, 1, 32, 64))
  sym_73427696  = model_const_tf(name="Rcnn_ctcV3/expand_conv1/conv2d_4/bias",shape=(64,))
  sym_223382672 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_6/kernel",shape=(1, 1, 64, 64))
  sym_356827536 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_6/bias",shape=(64,))
  sym_451228704 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/conv2d_5/kernel",shape=(3, 3, 32, 64))
  sym_88828560  = model_const_tf(name="Rcnn_ctcV3/expand_conv1/conv2d_5/bias",shape=(64,))
  sym_379167584 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/gamma",shape=(64,))
  sym_492256464 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/beta",shape=(64,))
  sym_104779696 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/moving_mean",shape=(64,))
  sym_378983504 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/moving_variance",shape=(64,))
  sym_73418512  = model_const_tf(name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/add/y",shape=(1,))
  sym_134609696 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_7/kernel",shape=(1, 1, 64, 64))
  sym_131967104 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/activation/conv2d_7/bias",shape=(64,))
  sym_473740336 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/activation/max_2/mul/x",shape=(1,))
  sym_112766576 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/conv2d_8/kernel",shape=(3, 3, 64, 64))
  sym_105635760 = model_const_tf(name="Rcnn_ctcV3/expand_conv1/conv2d_8/bias",shape=(64,))

  # sym_149080784 = tf.placeholder(shape=(1,108,21,32),dtype=tf.float32)

  sym_457698256 = tf.pad(sym_149080784,paddings=tf.constant(((0, 0), (0, 0), (0, 0), (0, 0))))
  #sym_382705136 = tf.conv2d(sym_457698256,sym_75044384,padding=[0, 0],dilation=(1, 1),layout="NHWC",strides=(1, 1),kernel_size=(1, 1),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/conv2d_4/convolution",use_bias=False)
  sym_382705136 = tf.nn.conv2d( sym_457698256,sym_75044384,
      padding="VALID",data_format="NHWC",strides=(1,1,1,1),name="Rcnn_ctcV3/expand_conv1/conv2d_4/convolution")

  sym_457698256 = tf.add(sym_382705136,sym_73427696)
  #sym_394053216 = tvm.pad(sym_149080784,pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))
  sym_394053216 = tf.pad(sym_149080784,paddings=tf.constant(((0, 0), (1, 1), (1, 1), (0, 0))))
  #sym_140340480 = tvm.conv2d(sym_394053216,sym_451228704,dilation=(1, 1),layout="NHWC",strides=(1, 1),padding=[0, 0],kernel_size=(3, 3),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/conv2d_5/convolution",use_bias=False)
  sym_140340480 = tf.nn.conv2d(sym_394053216,sym_451228704,data_format="NHWC",strides=(1,1,1,1),padding="VALID",name="Rcnn_ctcV3/expand_conv1/conv2d_5/convolution")
  sym_394053216 = tf.add(sym_140340480,sym_88828560)
  sym_88729488 = tf.add(sym_378983504,sym_73418512,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/add")
  sym_104808848 = tf.sqrt(sym_88729488,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/Rsqrt")
  sym_80975232 = tf.multiply(sym_104808848,sym_379167584,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/mul")
  sym_86811088 = tf.multiply(sym_394053216,sym_80975232,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/mul_1")
  sym_382126160 = tf.multiply(sym_104779696,sym_80975232,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/mul_2")
  sym_104808912 = tf.subtract(sym_492256464,sym_382126160,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/sub")
  sym_114622080 = tf.add(sym_86811088,sym_104808912,name="Rcnn_ctcV3/expand_conv1/static_batch_normalization_3/batchnorm/add_1")
  sym_382620512 = tf.pad(sym_114622080,paddings=tf.constant(((0,0),(0,0),(0,0),(0,0))))
  # sym_457310224 = tvm.conv2d(sym_382620512,sym_223382672,layout="NHWC",strides=(1, 1),padding=[0, 0],dilation=(1, 1),kernel_size=(1, 1),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/activation/conv2d_6/convolution",use_bias=False)
  sym_457310224 = tf.nn.conv2d(sym_382620512,sym_223382672,data_format="NHWC",strides=(1,1,1,1),padding="VALID",name="Rcnn_ctcV3/expand_conv1/activation/conv2d_6/convolution")
  sym_382620512 = tf.add(sym_457310224,sym_356827536)
  sym_74589536 = tf.pad(sym_114622080,paddings=tf.constant(((0,0),(0,0),(0,0),(0,0))))
  # sym_73915248 = tf.conv2d(sym_74589536,sym_134609696,layout="NHWC",strides=(1, 1),padding=[0, 0],dilation=(1, 1),kernel_size=(1, 1),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/activation/conv2d_7/convolution",use_bias=False)
  sym_73915248 = tf.nn.conv2d(sym_74589536,sym_134609696,data_format="NHWC",strides=(1,1,1,1),padding="VALID",name="Rcnn_ctcV3/expand_conv1/activation/conv2d_7/convolution")
  sym_74589536 = tf.add(sym_73915248,sym_131967104)
  sym_73915280 = tf.add(sym_382620512,sym_74589536,name="Rcnn_ctcV3/expand_conv1/activation/max_2/add")
  sym_73550304 = tf.subtract(sym_382620512,sym_74589536,name="Rcnn_ctcV3/expand_conv1/activation/max_2/sub")
  sym_457788432 = tf.nn.relu(sym_73550304,name="Rcnn_ctcV3/expand_conv1/activation/max_2/Relu")
  sym_457788880 = tf.add(sym_73915280,sym_457788432,name="Rcnn_ctcV3/expand_conv1/activation/max_2/add_1")
  sym_105411888 = tf.subtract(sym_74589536,sym_382620512,name="Rcnn_ctcV3/expand_conv1/activation/max_2/sub_1")
  sym_76993232 = tf.nn.relu(sym_105411888,name="Rcnn_ctcV3/expand_conv1/activation/max_2/Relu_1")
  sym_223578592 = tf.add(sym_457788880,sym_76993232,name="Rcnn_ctcV3/expand_conv1/activation/max_2/add_2")
  sym_152477424 = tf.multiply(sym_473740336,sym_223578592,name="Rcnn_ctcV3/expand_conv1/activation/max_2/mul")
  sym_393365024 = tf.pad(sym_152477424,paddings=tf.constant(((0,0),(1,1),(1,1),(0,0))))
  # sym_79064400 = tf.nn.conv2d(sym_393365024,sym_112766576,dilation=(1, 1),layout="NHWC",strides=(1,1),padding=[0,0],kernel_size=(3,3),channels=64,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/conv2d_8/convolution",use_bias=False)
  sym_79064400 = tf.nn.conv2d(sym_393365024,sym_112766576,data_format="NHWC",strides=(1,1,1,1),padding="VALID",name="Rcnn_ctcV3/expand_conv1/conv2d_8/convolution")
  sym_393365024 = tf.add(sym_79064400,sym_105635760)
  sym_118484944 = tf.add(sym_393365024,sym_457698256,name="Rcnn_ctcV3/expand_conv1/add_1/add")
  # End of Cell 1
  return sym_118484944


def _check1():
  na=np.ones(shape=(1,9,9,1))
  nb=np.array([1,1,1,1,3,1,1,1,1]).reshape((3,3,1,1))
  c1=with_nnvm(
      [na,nb],
      lambda a,b: sym.conv2d(a,b,
        padding=[0, 0],dilation=(1, 1),layout="NHWC",strides=(1, 1),kernel_size=(nb.shape[0], nb.shape[1]),
        channels=1,kernel_layout="HWIO",name="Rcnn_ctcV3/expand_conv1/conv2d_4/convolution",use_bias=False),
      ).reshape((1,-1))
  c2=with_tf(
      [na,nb],
      lambda a,b: tf.nn.conv2d(a,b,
        padding="VALID",data_format="NHWC",strides=(1,1,1,1),name="Rcnn_ctcV3/expand_conv1/conv2d_4/convolution"),
      ).reshape((1,-1))
  print(c1)
  print(c2)

def _check3():
  na=np.ones(shape=(1,9,9,1))
  c1=with_tf(
      [na],
      lambda a: tf.get_variable("test_var", shape=(1,1,1)))
  print(c1)

def check1_slice():
  na=np.ones(shape=(5,5))
  c1=with_nnvm([na], lambda a: sym.strided_slice(a,begin=(0,0),end=(5,3)))
  print(c1)


def model1_check_correctness():
  """ TODO: Results match very approximately. Increasing abs(na) leads to
  further decrease in precigion """

  na=0.8*np.ones(shape=(1,108,21,32))
  r1=with_tf(
      [na],
      lambda a: model1_block_tf(a))

  def m1(a):
    state=model1_block_nnvm_consts()
    return model1_block_nnvm(a,state)

  r2=with_nnvm(
      [na],
      lambda a: m1(a),
      MODEL1_BLOCK_PARAMS)

  # print(r1)
  # print(r2)
  np.testing.assert_allclose(r1, r2, atol=5e-1)
  print(r1.shape) # (1,108,21,64)


def model1_tvm(inp, nblocks:int=1):
  state=model1_block_nnvm_consts()
  prev=inp
  for n in range(nblocks):
    prev=model1_block_nnvm(prev,state)
    prev=sym.strided_slice(prev,begin=(0,0,0,0), end=inp.shape)
  return prev



# def check_model1():
  
