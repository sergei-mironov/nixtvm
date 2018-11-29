""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example ported from TensorFlow to TVM (to experiment with AutoTVM).
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author of TensorFlow impl: Aymeric Damien, https://github.com/aymericdamien/TensorFlow-Examples/
"""

# FIXME doesn't work yet
# Next missing method is add_variable/add_weight

import tvm
import topi
import numpy as np
import collections

from tensorflow.python.util import nest
from tensorflow.python.util import function_utils

# Import MNIST data
# FIXME uncomment when proxy is fixed
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 200  # FIXME: was 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

# Graph input
X = tvm.placeholder((batch_size, timesteps, num_input), dtype="float")
Y = tvm.placeholder((batch_size, num_classes), dtype="float")

# FIXME how to initialize var and specify shape? Are variables in TF/TVM the same at all?
# Define weights
weights = {
    'out': tvm.var("out_weights", dtype="float")  # tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tvm.var("out_bias", dtype="float")  # tf.Variable(tf.random_normal([num_classes]))
}


class LSTMCell:  # (LayerRNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.

    The default non-peephole implementation is based on:

      https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf

    Felix Gers, Jurgen Schmidhuber, and Fred Cummins.
    "Learning to forget: Continual prediction with LSTM." IET, 850-855, 1999.

    The peephole implementation is based on:

      https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.

    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.

    Note that this cell is not optimized for performance. Please use
    `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
    `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
    better performance on CPU.
    """

    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None, name=None, dtype=None, **kwargs):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`. It
            could also be string that is within Keras activation function names.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          dtype: Default dtype of the layer (default of `None` means use the type
            of the first input). Required when `build` is called before `call`.
          **kwargs: Dict, keyword named properties for common layer attributes, like
            `trainable` etc when constructing the cell from configs of get_config().

          When restoring from CudnnLSTM-trained checkpoints, use
          `CudnnCompatibleLSTMCell` instead.
        """
        # Inputs must be 2-dimensional.
        self.input_spec = InputSpec(ndim=2)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        # self._initializer = initializers.get(initializer)
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._dtype = dtype
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = topi.tanh

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units
        self.built = False

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    #   @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (
            partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units],
            initializer=self._initializer,
            partitioner=maybe_partitioner)
        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=initializer)
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                               initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
                if self._num_proj_shards is not None
                else None)
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)

        self.built = True

    def __call__(self, inputs, *args, **kwargs):
        """Wraps `call`, applying pre- and post-processing steps.

        Arguments:
          inputs: input tensor(s).
          *args: additional positional arguments to be passed to `self.call`.
          **kwargs: additional keyword arguments to be passed to `self.call`.
            **Note**: kwarg `scope` is reserved for use by the layer.

        Returns:
          Output tensor(s).

        Note:
          - If the layer's `call` method takes a `scope` keyword argument,
            this argument will be automatically set to the current variable scope.
          - If the layer's `call` method takes a `mask` argument (as some Keras
            layers do), its default value will be set to the mask generated
            for `inputs` by the previous layer (if `input` did come from
            a layer that generated a corresponding mask, i.e. if it came from
            a Keras layer with masking support.

        Raises:
          ValueError: if the layer's `call` method returns None (an invalid value).
        """

        # try:
        #     call_has_scope_arg = self._call_has_scope_arg
        # except AttributeError:
        #     self._call_fn_args = function_utils.fn_args(self.call)
        #     self._call_has_scope_arg = 'scope' in self._call_fn_args
        #     call_has_scope_arg = self._call_has_scope_arg
        # if call_has_scope_arg:
        #     kwargs['scope'] = scope

        # Actually call layer
        outputs = self.super__call__(inputs, *args, **kwargs)

        # Update global default collections.
        # _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)
        return outputs

    def _no_dependency(self, value):
        """Override to allow CheckpointableBase to disable dependency tracking."""
        return value # data_structures.NoDependency(value)

    def super__call__(self, inputs, *args, **kwargs):
        """Wraps `call`, applying pre- and post-processing steps.

        Arguments:
          inputs: input tensor(s).
          *args: additional positional arguments to be passed to `self.call`.
          **kwargs: additional keyword arguments to be passed to `self.call`.

        Returns:
          Output tensor(s).

        Note:
          - The following optional keyword arguments are reserved for specific uses:
            * `training`: Boolean scalar tensor of Python boolean indicating
              whether the `call` is meant for training or inference.
            * `mask`: Boolean input mask.
          - If the layer's `call` method takes a `mask` argument (as some Keras
            layers do), its default value will be set to the mask generated
            for `inputs` by the previous layer (if `input` did come from
            a layer that generated a corresponding mask, i.e. if it came from
            a Keras layer with masking support.

        Raises:
          ValueError: if the layer's `call` method returns None (an invalid value).
        """
        input_list = nest.flatten(inputs)

        build_graph = True
        # TODO(fchollet, allenl): Make deferred mode work with subclassed Models
        # which don't use an "inputs" argument.
        in_deferred_mode = False  # isinstance(input_list[0], DeferredTensor)

        # Handle Keras mask propagation from previous layer to current layer.
        previous_mask = None
        if build_graph and (not hasattr(self, '_compute_previous_mask') or
                            self._compute_previous_mask):
            previous_mask = collect_previous_mask(inputs)
            if not hasattr(self, '_call_fn_args'):
                self._call_fn_args = self._no_dependency(
                    function_utils.fn_args(self.call))
            if ('mask' in self._call_fn_args and 'mask' not in kwargs and
                    not generic_utils.is_all_none(previous_mask)):
                # The previous layer generated a mask, and mask was not explicitly pass
                # to __call__, hence we set previous_mask as the default value.
                kwargs['mask'] = previous_mask

        input_shapes = None

        if not self.built:
            if not build_graph:
                # Activity regularization is currently unsupported in Eager mode.
                if self._activity_regularizer:
                    raise ValueError(
                        'activity_regularizer currently unsupported with '
                        'eager execution enabled. Found an activity_regularizer in '
                        '%s(%s).' % (self.__class__.__name__, self))
            if not build_graph and not in_deferred_mode:
                for x in input_list:
                    if hasattr(x, '_keras_history'):
                        raise ValueError('_keras_history currently unsupported in '
                                         'Eager mode. Found _keras_history in %s while '
                                         'executing __call__ for %s(%s)' %
                                         (x, self.__class_.__name__, self))

            # Check input assumptions set before layer building, e.g. input rank.
            self._assert_input_compatibility(inputs)
            if input_list and self._dtype is None:
                try:
                    self._dtype = input_list[0].dtype.base_dtype.name
                except AttributeError:
                    pass

            if all(hasattr(x, 'shape') for x in input_list):
                input_shapes = nest.map_structure(lambda x: x.shape, inputs)

            if (not hasattr(self, '_is_graph_network') or
                    self.__class__.__name__ == 'Sequential' or
                    not hasattr(self.build, '_is_default')):
                # Only if self is a layer, an instance of a sequential model, or
                # the user has manually overwritten the build method do we need to
                # build it.
                self.build(input_shapes)
            # We must set self.built since user defined build functions are not
            # constrained to set self.built.
            self.built = True

        # Check input assumptions set after layer building, e.g. input shape.
        if build_graph or in_deferred_mode:
            self._assert_input_compatibility(inputs)

        if not in_deferred_mode:
            self._in_call = True
            outputs = self.call(inputs, *args, **kwargs)
            self._in_call = False
            if outputs is None:
                raise ValueError('A layer\'s `call` method should return a Tensor '
                                 'or a list of Tensors, not None (layer: ' +
                                 self.name + ').')
        else:
            # Deferred mode behavior: use `compute_output_shape` to
            # infer the number of outputs of the layer and their shapes.
            if input_shapes is None:
                input_shapes = nest.map_structure(lambda x: x.shape, inputs)

            output_shapes = self.compute_output_shape(input_shapes)
            output_shapes = nest.flatten(output_shapes)
            outputs = [
                # TODO(fchollet): name the deferred tensors?
                DeferredTensor(shape=shape, dtype=self._dtype)
                for shape in output_shapes
            ]
            if len(outputs) == 1:
                outputs = outputs[0]

        if build_graph:
            self._handle_activity_regularization(inputs, outputs)
            self._set_mask_metadata(inputs, outputs, previous_mask)

        if in_deferred_mode or build_graph and have_all_keras_metadata(inputs):
            inputs, outputs = self._set_connectivity_metadata_(
                inputs, outputs, args, kwargs)

        if hasattr(self, '_symbolic_set_inputs') and not self.inputs:
            # Subclassed network: explicitly set metadata normally set by a call to
            # self._set_inputs(). This is not relevant in eager execution.
            self._symbolic_set_inputs(inputs, outputs)

        if in_deferred_mode or build_graph:
            self._set_learning_phase_metadata(inputs, outputs)

        # Optionally load weight values that were specified at layer instantiation.
        # TODO(fchollet): consider enabling this with eager execution too.
        if hasattr(self, '_initial_weights') and self._initial_weights is not None:
            self.set_weights(self._initial_weights)
            del self._initial_weights
        return outputs

    def call(self, inputs, state):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, must be 2-D, `[batch, input_size]`.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.

        Returns:
          A tuple containing:

          - A `2-D, [batch, output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = topi.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = topi.slice(state, [0, 0], [-1, self._num_units])
            m_prev = topi.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.shape[1]  # inputs.shape.with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.shape[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = topi.matmul(
            topi.concatenate((inputs, m_prev), 1), self._kernel)
        lstm_matrix = topi.nn.bias_add(lstm_matrix, self._bias)

        i, j, f, o = topi.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            m = topi.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     topi.concatenate((c, m), 1))
        return m, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "use_peepholes": self._use_peepholes,
            "cell_clip": self._cell_clip,
            "initializer": initializers.serialize(self._initializer),
            "num_proj": self._num_proj,
            "proj_clip": self._proj_clip,
            "num_unit_shards": self._num_unit_shards,
            "num_proj_shards": self._num_proj_shards,
            "forget_bias": self._forget_bias,
            "state_is_tuple": self._state_is_tuple,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(LSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
          the shapes `[batch_size, s]` for each s in `state_size`.
        """
        # Try to use the last cached zero_state. This is done to avoid recreating
        # zeros, especially when eager execution is enabled.
        state_size = self.state_size
        if hasattr(self, "_last_zero_state"):
            (last_state_size, last_batch_size, last_dtype,
             last_output) = getattr(self, "_last_zero_state")
            if (last_batch_size == batch_size and
                    last_dtype == dtype and
                    last_state_size == state_size):
                return last_output

        output = _zero_state_tensors(state_size, batch_size, dtype)
        self._last_zero_state = (state_size, batch_size, dtype, output)
        return output

    def _assert_input_compatibility(self, inputs):
        """Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `inputs` verify the input assumptions
        of the layer (if any). If not, a clear and actional exception gets raised.

        Arguments:
            inputs: input tensor or list of input tensors.

        Raises:
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        """
        if not self.input_spec:
            return
        if not isinstance(self.input_spec, (list, tuple)):
            input_spec = nest.flatten(self.input_spec)
        else:
            input_spec = self.input_spec
        inputs = nest.flatten(inputs)
        if len(inputs) != len(input_spec):
            raise ValueError('Layer ' + self.name + ' expects ' +
                             str(len(input_spec)) + ' inputs, '
                                                    'but it received ' + str(len(inputs)) +
                             ' input tensors. Inputs received: ' + str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
            if spec is None:
                continue

            # Check ndim.
            if spec.ndim is not None:
                ndim = len(x.shape)
                if ndim != spec.ndim:
                    raise ValueError('Input ' + str(input_index) + ' of layer ' +
                                     self.name + ' is incompatible with the layer: '
                                                 'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                                     str(ndim) + '. Full shape received: ' +
                                     str(x.shape.as_list()))
            if spec.max_ndim is not None:
                ndim = len(x.shape)
                if ndim is not None and ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) + ' of layer ' +
                                     self.name + ' is incompatible with the layer: '
                                                 'expected max_ndim=' + str(spec.max_ndim) +
                                     ', found ndim=' + str(ndim))
            if spec.min_ndim is not None:
                ndim = len(x.shape)
                if ndim is not None and ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) + ' of layer ' +
                                     self.name + ' is incompatible with the layer: '
                                                 ': expected min_ndim=' + str(spec.min_ndim) +
                                     ', found ndim=' + str(ndim) +
                                     '. Full shape received: ' +
                                     str(x.shape.as_list()))
            # Check dtype.
            if spec.dtype is not None:
                if x.dtype != spec.dtype:
                    raise ValueError('Input ' + str(input_index) + ' of layer ' +
                                     self.name + ' is incompatible with the layer: '
                                                 'expected dtype=' + str(spec.dtype) +
                                     ', found dtype=' + str(x.dtype))
            # Check specific shape axes.
            if spec.axes:
                shape = x.shape.as_list()
                if shape is not None:
                    for axis, value in spec.axes.items():
                        if hasattr(value, 'value'):
                            value = value.value
                        if value is not None and shape[int(axis)] not in {value, None}:
                            raise ValueError(
                                'Input ' + str(input_index) + ' of layer ' + self.name + ' is'
                                                                                         ' incompatible with the layer: expected axis ' + str(
                                    axis) +
                                ' of input shape to have value ' + str(value) +
                                ' but received input with shape ' + str(shape))
            # Check shape.
            if spec.shape is not None:
                shape = x.shape.as_list()
                if shape is not None:
                    for spec_dim, dim in zip(spec.shape, shape):
                        if spec_dim is not None and dim is not None:
                            if spec_dim != dim:
                                raise ValueError('Input ' + str(input_index) +
                                                 ' is incompatible with layer ' + self.name +
                                                 ': expected shape=' + str(spec.shape) +
                                                 ', found shape=' + str(shape))

def collect_previous_mask(input_tensors):
  """Retrieves the output mask(s) of the previous node.

  Arguments:
      input_tensors: A tensor or list of tensors.

  Returns:
      A mask tensor or list of mask tensors.
  """
  input_tensors = nest.flatten(input_tensors)
  masks = []
  for x in input_tensors:
    if hasattr(x, '_keras_mask'):
      mask = x._keras_mask  # pylint: disable=protected-access
      masks.append(mask)
    else:
      masks.append(None)
  if len(masks) == 1:
    return masks[0]
  return masks

def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""

    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = (batch_size, s) if isinstance(s, int) else (batch_size,) + s
        dtype1 = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        zero = dtype1.type(0)
        zeros = tvm.compute(c, lambda *args: zero, name="zeros")  # tvm.ndarray.array(np.zeros(c, dtype=dtype))
        return zeros

    return nest.map_structure(get_state_shape, state_size)


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

    Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
    and `h` is the output.

    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype

class InputSpec(object):
    """Specifies the ndim, dtype and shape of every input to a layer.

    Every layer should expose (if appropriate) an `input_spec` attribute:
    a list of instances of InputSpec (one per input tensor).

    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.

    Arguments:
        dtype: Expected DataType of the input.
        shape: Shape tuple, expected shape of the input
            (may include None for unchecked axes).
        ndim: Integer, expected rank of the input.
        max_ndim: Integer, maximum rank of the input.
        min_ndim: Integer, minimum rank of the input.
        axes: Dictionary mapping integer axes to
            a specific dimension value.
    """

    def __init__(self,
                 dtype=None,
                 shape=None,
                 ndim=None,
                 max_ndim=None,
                 min_ndim=None,
                 axes=None):
        self.dtype = dtype
        self.shape = shape
        if shape is not None:
            self.ndim = len(shape)
        else:
            self.ndim = ndim
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.axes = axes or {}

    def __repr__(self):
        spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
                ('shape=' + str(self.shape)) if self.shape else '',
                ('ndim=' + str(self.ndim)) if self.ndim else '',
                ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
                ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
                ('axes=' + str(self.axes)) if self.axes else '']
        return 'InputSpec(%s)' % ', '.join(x for x in spec if x)

def static_rnn(cell,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    The simplest form of RNN network generated is:

    ```python
        state = cell.zero_state(...)
        outputs = []
        for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
        return (outputs, state)
    ```
    However, a few other options are available:

    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.

    The dynamic calculation performed is, at time `t` for batch row `b`,

    ```python
        (output, state)(b, t) =
        (t >= sequence_length(b))
            ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
            : cell(input(b, t), state(b, t - 1))
    ```

    Args:
        cell: An instance of RNNCell.
        inputs: A length T list of inputs, each a `Tensor` of shape
        `[batch_size, input_size]`, or a nested tuple of such elements.
        initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
        sequence_length: Specifies the length of each sequence in inputs.
        An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
        scope: VariableScope for the created subgraph; defaults to "rnn".

    Returns:
        A pair (outputs, state) where:

        - outputs is a length T list of outputs (one for each input), or a nested
        tuple of such elements.
        - state is the final state

    Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If `inputs` is `None` or an empty list, or if the input depth
        (column size) cannot be inferred from inputs via shape inference.
    """
    state = cell.zero_state(batch_size, dtype)
    outputs = []
    for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
    return (outputs, state)


#   if not nest.is_sequence(inputs):
#     raise TypeError("inputs must be a sequence")
#   if not inputs:
#     raise ValueError("inputs must not be empty")

#   outputs = []
#   # Create a new scope in which the caching device is either
#   # determined by the parent scope, or is set to place the cached
#   # Variable using the same placement as for the rest of the RNN.
#   with vs.variable_scope(scope or "rnn") as varscope:
#     if _should_cache():
#       if varscope.caching_device is None:
#         varscope.set_caching_device(lambda op: op.device)

#     # Obtain the first sequence of the input
#     first_input = inputs
#     while nest.is_sequence(first_input):
#       first_input = first_input[0]

#     # Temporarily avoid EmbeddingWrapper and seq2seq badness
#     # TODO(lukaszkaiser): remove EmbeddingWrapper
#     if len(first_input.shape) != 1:

#       input_shape = first_input.shape #.with_rank_at_least(2)
#       fixed_batch_size = input_shape[0]

#       flat_inputs = nest.flatten(inputs)
#       for flat_input in flat_inputs:
#         input_shape = flat_input.shape #.with_rank_at_least(2)
#         batch_size, input_size = input_shape[0], input_shape[1:]
#         fixed_batch_size.merge_with(batch_size)
#         for i, size in enumerate(input_size):
#           if size.value is None:
#             raise ValueError(
#                 "Input size (dimension %d of inputs) must be accessible via "
#                 "shape inference, but saw value None." % i)
#     else:
#       fixed_batch_size = first_input.shape #.with_rank_at_least(1)[0]

#     if fixed_batch_size.value:
#       batch_size = fixed_batch_size.value
#     else:
#       batch_size = array_ops.shape(first_input)[0]
#     if initial_state is not None:
#       state = initial_state
#     else:
#       if not dtype:
#         raise ValueError("If no initial_state is provided, "
#                          "dtype must be specified")
#       if getattr(cell, "get_initial_state", None) is not None:
#         state = cell.get_initial_state(
#             inputs=None, batch_size=batch_size, dtype=dtype)
#       else:
#         state = cell.zero_state(batch_size, dtype)

#     if sequence_length is not None:  # Prepare variables
#       sequence_length = ops.convert_to_tensor(
#           sequence_length, name="sequence_length")
#       if sequence_length.ndim not in (None, 1):
#         raise ValueError(
#             "sequence_length must be a vector of length batch_size")

#       def _create_zero_output(output_size):
#         # convert int to TensorShape if necessary
#         size = _concat(batch_size, output_size)
#         output = array_ops.zeros(
#             array_ops.stack(size), _infer_state_dtype(dtype, state))
#         shape = _concat(fixed_batch_size.value, output_size, static=True)
#         output.set_shape(tensor_shape.TensorShape(shape))
#         return output

#       output_size = cell.output_size
#       flat_output_size = nest.flatten(output_size)
#       flat_zero_output = tuple(
#           _create_zero_output(size) for size in flat_output_size)
#       zero_output = nest.pack_sequence_as(
#           structure=output_size, flat_sequence=flat_zero_output)

#       sequence_length = topi.to_int32(sequence_length)
#       min_sequence_length = topi.reduce_min(sequence_length)
#       max_sequence_length = topi.reduce_max(sequence_length)

#     # Keras RNN cells only accept state as list, even if it's a single tensor.
#     is_keras_rnn_cell = _is_keras_rnn_cell(cell)
#     if is_keras_rnn_cell and not nest.is_sequence(state):
#       state = [state]
#     for time, input_ in enumerate(inputs):
#       if time > 0:
#         varscope.reuse_variables()
#       # pylint: disable=cell-var-from-loop
#       call_cell = lambda: cell(input_, state)
#       # pylint: enable=cell-var-from-loop
#       if sequence_length is not None:
#         (output, state) = _rnn_step(
#             time=time,
#             sequence_length=sequence_length,
#             min_sequence_length=min_sequence_length,
#             max_sequence_length=max_sequence_length,
#             zero_output=zero_output,
#             state=state,
#             call_cell=call_cell,
#             state_size=cell.state_size)
#       else:
#         (output, state) = call_cell()
#       outputs.append(output)
#     # Keras RNN cells only return state as list, even if it's a single tensor.
#     if is_keras_rnn_cell and len(state) == 1:
#       state = state[0]

#     return (outputs, state)

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = [topi.squeeze(x_i, axis=1) for x_i in topi.split(x, timesteps, 1)]

    # Define a lstm cell
    lstm_cell = LSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = static_rnn(lstm_cell, x, dtype="float32")

    # Linear activation, using rnn inner loop last output
    return topi.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = topi.nn.softmax(logits, name='prediction')

# Define loss and optimizer
loss_op = topi.reduce_mean(topi.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    print("Saving the model")
    simple_save(sess, export_dir='./saved_recurrent_network', inputs={"images": X}, outputs={"out": prediction})
