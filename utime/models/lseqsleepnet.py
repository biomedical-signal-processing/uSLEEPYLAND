"""
Implementation of L-SeqSleepNet model, as described in the paper:

Phan, Huy, et al. "L-SeqSleepNet: Whole-cycle long sequence
modelling for automatic sleep staging."
IEEE Journal of Biomedical and Health Informatics (2023).

The following is an adaptation of
https://github.com/pquochuy/l-seqsleepnet/tree/main
in the utime framework.
"""

import logging
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Layer, Dense, LayerNormalization,
                                     Dropout, Lambda, Input,
                                     Bidirectional, LSTM
                                     )
import numpy as np

logger = logging.getLogger(__name__)

@tf.keras.saving.register_keras_serializable(package="Custom", name="FilterbankLayer")
class FilterbankLayer(tf.keras.layers.Layer):
    def __init__(self,
                 nfilter,
                 nfft,
                 samplerate,
                 lowfreq,
                 highfreq,
                 freq_bins,
                 **kwargs):
        super(FilterbankLayer, self).__init__(**kwargs)
        self.nfilter = nfilter
        self.nfft = nfft
        self.samplerate = samplerate
        self.lowfreq = lowfreq
        self.highfreq = highfreq
        assert self.highfreq <= self.samplerate / 2, "highfreq is greater than samplerate/2"
        self.freq_bins = freq_bins
        self.filterbank_matrix = self.compute_filterbank_matrix()

    def compute_filterbank_matrix(self):
        """Compute the linear filterbank matrix."""
        highfreq = self.highfreq or self.samplerate / 2
        hzpoints = np.linspace(self.lowfreq, highfreq, self.nfilter + 2)
        bin = np.floor((self.nfft + 1) * hzpoints / self.samplerate).astype(int)

        fbank = np.zeros([self.nfilter, self.nfft // 2 + 1])
        for j in range(0, self.nfilter):
            for i in range(int(bin[j]), int(bin[j + 1])):
                fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in range(int(bin[j + 1]), int(bin[j + 2])):
                fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        fbank = np.transpose(fbank)
        fbank.astype(np.float32)
        return tf.constant(fbank, dtype=tf.float32)

    def build(self, input_shape):
        # Weights for EEG and EOG channels
        self.Weeg = self.add_weight(
            shape=(self.freq_bins, self.nfilter),
            initializer=tf.random_normal_initializer(stddev=1.0),
            trainable=True,
        )

        self.Weog = self.add_weight(
            shape=(self.freq_bins, self.nfilter),
            initializer=tf.random_normal_initializer(stddev=1.0),
            trainable=True,
        )

    def call(self, inputs):
        # Assuming inputs is of shape (B*L*T, freq_bins, channels)
        X_eeg = tf.reshape(tf.squeeze(inputs[:, :, 0]), [-1, self.freq_bins])
        Weeg_sigmoid = tf.sigmoid(self.Weeg)
        Wfb_eeg = tf.multiply(Weeg_sigmoid, self.filterbank_matrix)
        HWeeg = tf.matmul(X_eeg, Wfb_eeg)

        X_eog = tf.reshape(tf.squeeze(inputs[:, :, 1]), [-1, self.freq_bins])
        Weog_sigmoid = tf.sigmoid(self.Weog)
        Wfb_eog = tf.multiply(Weog_sigmoid, self.filterbank_matrix)
        HWeog = tf.matmul(X_eog, Wfb_eog)

        return tf.concat([HWeeg, HWeog], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'nfilter': self.nfilter,
            'nfft': self.nfft,
            'samplerate': self.samplerate,
            'lowfreq': self.lowfreq,
            'highfreq': self.highfreq,
            'freq_bins': self.freq_bins,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="Custom", name="RecurrentBatchNorm")
class RecurrentBatchNorm(tf.keras.layers.Layer):

    def __init__(self,
                 size,
                 max_bn_steps,
                 initial_scale=0.1,
                 decay=0.95,
                 no_offset=False,
                 set_forget_gate_bias=False,
                 regularizer=None,
                 epsilon=1e-5,
                 **kwargs):
        super(RecurrentBatchNorm, self).__init__(**kwargs)
        self.size = size
        self.max_bn_steps = max_bn_steps
        self.initial_scale = initial_scale
        self.decay = decay
        self.no_offset = no_offset
        self.set_forget_gate_bias = set_forget_gate_bias
        self.regularizer = regularizer
        self.epsilon = epsilon

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'max_bn_steps': self.max_bn_steps,
            'initial_scale': self.initial_scale,
            'decay': self.decay,
            'no_offset': self.no_offset,
            'set_forget_gate_bias': self.set_forget_gate_bias,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer),
            'epsilon': self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["regularizer"] = tf.keras.regularizers.deserialize(config["regularizer"])
        return cls(**config)

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=[self.size],
            initializer=tf.constant_initializer(self.initial_scale),
            trainable=True,
            regularizer=self.regularizer
        )
        if not self.no_offset:
            if self.set_forget_gate_bias:
                self.offset = self.add_weight(
                    shape=[self.size],
                    initializer=self.offset_initializer(),
                    trainable=True,
                    regularizer=self.regularizer
                )
            else:
                self.offset = self.add_weight(
                    shape=[self.size],
                    initializer=tf.zeros_initializer(),
                    trainable=True,
                    regularizer=self.regularizer
                )
        self.pop_mean_all_steps = self.add_weight(
            shape=[self.max_bn_steps, self.size],
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        self.pop_var_all_steps = self.add_weight(
            shape=[self.max_bn_steps, self.size],
            initializer=tf.ones_initializer(),
            trainable=False
        )
        self.built = True

    def offset_initializer(self):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            size = shape[0]
            assert size % 4 == 0
            size = size // 4
            res = [np.ones((size)), np.zeros((size * 3))]
            return tf.constant(np.concatenate(res, axis=0), dtype)

        return _initializer

    def call(self, x, step, training):
        # Convert the training argument to a TensorFlow boolean tensor
        is_training_tensor = tf.convert_to_tensor(training, dtype=tf.bool) if training is not None else tf.constant(
            False)

        # Ensure step is treated as a valid timestep value
        step = tf.minimum(step, self.max_bn_steps - 1)

        # Use tf.gather to get batch normalization statistics for each step in the batch
        pop_mean = tf.gather(self.pop_mean_all_steps, step)
        pop_var = tf.gather(self.pop_var_all_steps, step)

        batch_mean, batch_var = tf.nn.moments(x, [0])

        def batch_statistics():
            pop_mean_new = self.decay * pop_mean + (1 - self.decay) * batch_mean
            pop_var_new = self.decay * pop_var + (1 - self.decay) * batch_var

            # Update population statistics
            updated_pop_mean = tf.tensor_scatter_nd_update(self.pop_mean_all_steps,
                                                           tf.expand_dims(step, axis=-1),
                                                           pop_mean_new)
            updated_pop_var = tf.tensor_scatter_nd_update(self.pop_var_all_steps,
                                                          tf.expand_dims(step, axis=-1),
                                                          pop_var_new)

            # Assign the updated values to the original variables
            self.pop_mean_all_steps.assign(updated_pop_mean)
            self.pop_var_all_steps.assign(updated_pop_var)

            return tf.nn.batch_normalization(x=x,
                                             mean=batch_mean,
                                             variance=batch_var,
                                             offset=self.offset if not self.no_offset else None,
                                             scale=self.scale,
                                             variance_epsilon=self.epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x=x,
                                             mean=pop_mean,
                                             variance=pop_var,
                                             offset=self.offset if not self.no_offset else None,
                                             scale=self.scale,
                                             variance_epsilon=self.epsilon)

        return tf.cond(is_training_tensor, batch_statistics, population_statistics)


@tf.keras.saving.register_keras_serializable(package="Custom", name="BNLSTMCell")
class BNLSTMCell(tf.keras.layers.Layer):
    '''
    Batch normalized LSTM as described in arxiv.org/abs/1603.09025
    Translated to tf 2 from https://github.com/pquochuy/l-seqsleepnet/blob/main/sleepedf-20/network/lseqsleepnet/bnlstm.py#L6'''

    def __init__(
            self,
            num_units,
            max_bn_steps,  # max number of steps for which to store separate population stats
            initial_scale=0.1,
            activation=tf.nn.tanh,
            decay=0.95,
            input_keep_prob=1.0,
            output_keep_prob=1.0,
            regularizer=None,
            **kwargs
    ):

        super().__init__(**kwargs)
        self._num_units = num_units
        self._max_bn_steps = max_bn_steps
        self._activation = activation
        self._decay = decay
        self._initial_scale = initial_scale
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.step = None
        self.regularizer = regularizer

        self.bn_layers = []

    @property
    def state_size(self):
        return (self._num_units, self._num_units, 1)

    @property
    def output_size(self):
        return self._num_units

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_units': self._num_units,
            'max_bn_steps': self._max_bn_steps,
            'initial_scale': self._initial_scale,
            'activation': tf.keras.activations.serialize(self._activation),
            'decay': self._decay,
            'input_keep_prob': self.input_keep_prob,
            'output_keep_prob': self.output_keep_prob,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["activation"] = tf.keras.activations.deserialize(config["activation"])
        config["regularizer"] = tf.keras.regularizers.deserialize(config["regularizer"])
        return cls(**config)

    def orthogonal_lstm_initializer(self):
        def orthogonal(shape, dtype=tf.float32, partition_info=None):
            # taken from https://github.com/cooijmanstim/recurrent-batch-normalization
            # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
            """ benanne lasagne ortho init (faster than qr approach)"""
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shape
            q = q.reshape(shape)
            return tf.constant(q[:shape[0], :shape[1]], dtype)

        return orthogonal

    def build(self, input_shape):

        x_size = input_shape[-1]
        self.W_xh = self.add_weight(
            shape=(x_size, 4 * self._num_units),
            initializer=self.orthogonal_lstm_initializer(),
            regularizer=self.regularizer)

        self.W_hh = self.add_weight(
            shape=(self._num_units, 4 * self._num_units),
            initializer=self.orthogonal_lstm_initializer(),
            regularizer=self.regularizer)

        # bn for hh
        self.bn_layers.append(RecurrentBatchNorm(
            size=4 * self._num_units,
            max_bn_steps=self._max_bn_steps,
            initial_scale=self._initial_scale,
            decay=self._decay,
            set_forget_gate_bias=True,
            regularizer=self.regularizer,
            epsilon=1e-5
        ))

        # bn for xh
        self.bn_layers.append(RecurrentBatchNorm(
            size=4 * self._num_units,
            max_bn_steps=self._max_bn_steps,
            initial_scale=self._initial_scale,
            decay=self._decay,
            no_offset=True,  # no offset for xh
            regularizer=self.regularizer,
            epsilon=1e-5
        ))

        # bn for cell state
        self.bn_layers.append(RecurrentBatchNorm(
            size=self._num_units,
            max_bn_steps=self._max_bn_steps,
            initial_scale=self._initial_scale,
            decay=self._decay,
            regularizer=self.regularizer,
            epsilon=1e-5
        ))

        self.built = True

    def call(self, inputs, states, training=None):
        c, h, step = states

        if step is None:
            step = tf.Variable(0, trainable=False)  # Initialize step counter

        _step = tf.squeeze(tf.gather(tf.cast(step, tf.int32), 0))

        if training and self.input_keep_prob < 1.0:
            inputs = Dropout(rate=1 - self.input_keep_prob)(inputs, training=training)

        xh = tf.matmul(inputs, self.W_xh)  # (B, 4 * num_units)
        hh = tf.matmul(h, self.W_hh)  # (B, 4 * num_units)

        bn_hh = self.bn_layers[0](hh, _step, training)
        bn_xh = self.bn_layers[1](xh, _step, training)

        hidden = bn_xh + bn_hh

        # split into gates (forget, input, output, candidate)
        f, i, o, j = tf.split(hidden, 4, axis=-1)

        new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * self._activation(j)  # cell state
        bn_new_c = self.bn_layers[2](new_c, _step, training)
        new_h = self._activation(bn_new_c) * tf.sigmoid(o)

        # Apply dropout to the output if training
        if training and self.output_keep_prob < 1.0:
            new_h = Dropout(rate=1 - self.output_keep_prob)(new_h, training=training)

        return new_h, (new_c, new_h, step + 1)  # output, new state


@tf.keras.saving.register_keras_serializable(package="Custom", name="BidirectionalRNNLayerBN")
class BidirectionalRNNLayerBN(tf.keras.layers.Layer):
    def __init__(self,
                 nhidden,
                 nlayer,
                 seq_len=1,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 regularizer=None,
                 **kwargs):
        super(BidirectionalRNNLayerBN, self).__init__(**kwargs)
        self.nhidden = nhidden
        self.nlayer = nlayer
        self.seq_len = seq_len
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.regularizer = regularizer

        self.fw_cell = None
        self.bw_cell = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'nhidden': self.nhidden,
            'nlayer': self.nlayer,
            'seq_len': self.seq_len,
            'input_keep_prob': self.input_keep_prob,
            'output_keep_prob': self.output_keep_prob,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["regularizer"] = tf.keras.regularizers.deserialize(config["regularizer"])
        return cls(**config)

    def build(self, input_shape):
        # Building the forward and backward cells based on the number of layers
        def build_bn_lstm_cell():
            return BNLSTMCell(
                num_units=self.nhidden,
                max_bn_steps=self.seq_len,
                input_keep_prob=self.input_keep_prob,
                output_keep_prob=self.output_keep_prob,
                regularizer=self.regularizer
            )

        if self.nlayer == 1:
            self.fw_cell = build_bn_lstm_cell()
            self.bw_cell = build_bn_lstm_cell()
        else:
            self.fw_cell = tf.keras.layers.StackedRNNCells([build_bn_lstm_cell() for _ in range(self.nlayer)])
            self.bw_cell = tf.keras.layers.StackedRNNCells([build_bn_lstm_cell() for _ in range(self.nlayer)])

        super(BidirectionalRNNLayerBN, self).build(input_shape)

    def call(self, inputs, training=None):
        bi_rnn_layer = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.RNN(self.fw_cell,
                                      return_sequences=True,
                                      return_state=True,
                                      go_backwards=False),
            backward_layer=tf.keras.layers.RNN(self.bw_cell,
                                               return_sequences=True,
                                               return_state=True,
                                               go_backwards=True),
            merge_mode='concat'
        )

        rnn_outputs = bi_rnn_layer(inputs, training=training)

        outputs = rnn_outputs[0]
        fw_state = rnn_outputs[1:3]  # Forward state (c, h)
        bw_state = rnn_outputs[3:5]  # Backward state (c, h)

        return outputs, (fw_state, bw_state)

    def get_config(self):
        config = super(BidirectionalRNNLayerBN, self).get_config()
        config.update({
            'nhidden': self.nhidden,
            'nlayer': self.nlayer,
            'seq_len': self.seq_len,
            'is_training': self.is_training,
            'input_keep_prob': self.input_keep_prob,
            'output_keep_prob': self.output_keep_prob,
            'regularizer': self.regularizer
        })
        return config


@tf.keras.saving.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size,
                 attention_size,
                 regularizer=None):
        super(AttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.regularizer = regularizer
        self.W_omega = self.add_weight(shape=(hidden_size, attention_size),
                                       initializer=tf.random_normal_initializer(stddev=0.1),
                                       regularizer=regularizer)
        self.b_omega = self.add_weight(shape=(attention_size,),
                                       initializer=tf.random_normal_initializer(stddev=0.1),
                                       regularizer=regularizer)
        self.u_omega = self.add_weight(shape=(attention_size,),
                                       initializer=tf.random_normal_initializer(stddev=0.1),
                                       regularizer=regularizer)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'attention_size': self.attention_size,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["regularizer"] = tf.keras.regularizers.deserialize(config["regularizer"])
        return cls(**config)

    def call(self, inputs):
        # inputs have shape (B*L, T, hidden_size)
        # this is essentially just a dense layer with tanh
        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, self.hidden_size]),
                              self.W_omega) + self.b_omega)  # (B*L*T, attention_size)

        # multiply by epoch-level context vector
        vu = tf.matmul(v, tf.reshape(self.u_omega, [-1, 1]))  # (B*L*T, 1)

        # Reshape to [batch_size, sequence_length] and apply softmax to get the attention weights
        exps = tf.reshape(tf.exp(vu), [-1, inputs.shape[1]])  # (B*L, T)
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  # (B*L, T)

        # Compute the epoch representation as a weighted sum
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, inputs.shape[1], 1]), 1)  # (B*L, hidden_size)

        return output


@tf.keras.saving.register_keras_serializable(package="Custom", name="SequenceEncoder")
class SequenceEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 nhidden_seq,
                 lstm_nlayer_seq,
                 nsubseq,
                 sub_seq_len,
                 dual_rnn_blocks,
                 dropout_rnn,
                 regularizer=None,
                 **kwargs):
        super(SequenceEncoder, self).__init__(**kwargs)
        self.nhidden_seq = nhidden_seq
        self.lstm_nlayer_seq = lstm_nlayer_seq
        self.nsubseq = nsubseq
        self.sub_seq_len = sub_seq_len
        self.dual_rnn_blocks = dual_rnn_blocks
        self.dropout_rnn = dropout_rnn
        self.regularizer = regularizer

    def get_config(self):
        config = super().get_config()
        config.update({
            'nhidden_seq': self.nhidden_seq,
            'lstm_nlayer_seq': self.lstm_nlayer_seq,
            'nsubseq': self.nsubseq,
            'sub_seq_len': self.sub_seq_len,
            'dual_rnn_blocks': self.dual_rnn_blocks,
            'dropout_rnn': self.dropout_rnn,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["regularizer"] = tf.keras.regularizers.deserialize(config["regularizer"])
        return cls(**config)

    def build(self, input_shape):
        self.rnn_cells_horizontal = [
            BidirectionalRNNLayerBN(
                nhidden=self.nhidden_seq,
                nlayer=self.lstm_nlayer_seq,
                seq_len=self.sub_seq_len,
                input_keep_prob=1.0 if block == 0 else self.dropout_rnn,
                output_keep_prob=self.dropout_rnn,
                regularizer=self.regularizer
            ) for block in range(self.dual_rnn_blocks)
        ]

        self.rnn_cells_vertical = [
            BidirectionalRNNLayerBN(
                nhidden=self.nhidden_seq,
                nlayer=self.lstm_nlayer_seq,
                seq_len=self.nsubseq,
                input_keep_prob=self.dropout_rnn,
                output_keep_prob=self.dropout_rnn,
                regularizer=self.regularizer
            ) for _ in range(self.dual_rnn_blocks)
        ]

        super(SequenceEncoder, self).build(input_shape)

    def residual_rnn(self, input_tensor, rnn_layer, seq_len, training):
        # Process the input through the bidirectional RNN
        rnn_output, _ = rnn_layer(input_tensor, training=training)

        # Flatten and project the output using a Dense layer, then normalize
        rnn_output_flat = tf.reshape(rnn_output, [-1, self.nhidden_seq * 2])
        projected = tf.keras.layers.Dense(units=input_tensor.shape[-1],
                                          activation=None,
                                          kernel_regularizer=self.regularizer)(rnn_output_flat)
        normalized = tf.keras.layers.LayerNormalization(
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer)(projected)
        rnn_output_reshaped = tf.reshape(normalized, [-1, seq_len, input_tensor.shape[-1]])

        # Add residual connection
        return rnn_output_reshaped + input_tensor

    def call(self, inputs, training=None):
        # Process the input for dual-sequence modeling
        batch_size, nsubseq, subseq_len, dim = inputs.shape.as_list()

        for block in range(self.dual_rnn_blocks):
            # Horizontal sequence modeling (process all subsequences in parallel)
            inputs = tf.reshape(inputs, [-1, subseq_len, dim])  # [B*L_sub, K, dim]
            inputs = self.residual_rnn(
                inputs,
                self.rnn_cells_horizontal[block],
                self.sub_seq_len,
                training=training)

            # Reshape to handle vertical sequence modeling
            inputs = tf.reshape(inputs, [-1, nsubseq, subseq_len, dim])  # [B, L_sub, K, dim]
            inputs = tf.transpose(inputs, [0, 2, 1, 3])  # [B, K, L_sub, dim]
            inputs = tf.reshape(inputs, [-1, nsubseq, dim])  # [B*K, L_sub, dim]
            inputs = self.residual_rnn(
                inputs,
                self.rnn_cells_vertical[block],
                self.nsubseq,
                training=training)

            # Reshape back to the original format
            inputs = tf.reshape(inputs, [-1, subseq_len, nsubseq, dim])  # [B, K, L_sub, dim]
            inputs = tf.transpose(inputs, [0, 2, 1, 3])  # Transpose back to [B, L_sub, K, dim]

        # Final reshaping after the dual-sequence encoding
        inputs = tf.reshape(inputs, [-1, nsubseq, subseq_len, dim])  # [B, L_sub, K, dim]
        inputs = Dropout(rate=1 - self.dropout_rnn)(inputs, training=training)

        return inputs


@tf.keras.saving.register_keras_serializable(package="Custom", name="LSeqSleepNet")
class LSeqSleepNet(Model):

    def __init__(self,
                 n_classes,
                 batch_shape,
                 nsubseq,
                 nfilter,
                 nfft,
                 samplerate,
                 lowfreq,
                 highfreq,
                 nhidden_epoch,
                 nhidden_seq,
                 lstm_nlayer_epoch,
                 lstm_nlayer_seq,
                 dual_rnn_blocks,
                 dropout_rnn,
                 attention_size,
                 fc_size,
                 l2_reg=None,
                 # build=True,
                 no_log=False,
                 **kwargs
                 ):
        """

        Args:
            n_classes (int):
                Number of classes to predict
            batch_shape (tuple):
                Shape of the input batch (B, L, T, F, C)
            nsubseq (int):
                Number of subsequences to divide the sequence into
            nfilter (int):
                Number of filters for the filterbank
            nfft (int):
                Number of FFT points
            samplerate (int):
                Sampling rate of the input signal
            lowfreq (int):
                Lower frequency bound for the filterbank
            highfreq (int):
                Upper frequency bound for the filterbank
            nhidden_epoch (int):
                Number of hidden units for the epoch encoder
            nhidden_seq (int):
                Number of hidden units for the sequence encoder
            lstm_nlayer_epoch (int):
                Number of LSTM layers for the epoch encoder
            lstm_nlayer_seq (int):
                Number of LSTM layers for the sequence encoder
            dual_rnn_blocks (int):
                Number of dual RNN blocks for the sequence encoder
            dropout_rnn (float):
                Dropout rate for the RNN layers
            attention_size (int):
                Size of the attention layer
            fc_size (int):
                Size of the fully connected layers in the output classifier
            l2_reg (float):
                L2 regularization parameter, default None
            no_log (bool):
                Whether to not log the model, default False
        """

        super(LSeqSleepNet, self).__init__()
        self.model_name = "LSeqSleepNet"

        # input and output shapes
        assert len(batch_shape) == 5  # (B, L, T, F, C)
        self.n_classes = n_classes
        self.batch_shape = batch_shape
        self.epoch_seq_len = batch_shape[1]  # L
        self.frame_seq_len = batch_shape[2]  # T
        self.freq_bins = batch_shape[3]  # F, ndim
        self.channels = batch_shape[4]  # C

        # params for filterbank
        self.nfilter = nfilter
        self.nfft = nfft
        self.samplerate = samplerate
        self.lowfreq = lowfreq
        self.highfreq = highfreq

        # Epoch encoder params
        self.nhidden_epoch = nhidden_epoch
        self.lstm_nlayer_epoch = lstm_nlayer_epoch

        # Attention layer params
        self.attention_size = attention_size

        # Sequence encoder params
        self.nsubseq = nsubseq  # In the paper this is B but we call it L_sub to avoid confusion
        assert self.epoch_seq_len % self.nsubseq == 0  # L % L_sub == 0
        self.sub_seq_len = self.epoch_seq_len // self.nsubseq  # K in the paper
        self.nhidden_seq = nhidden_seq
        self.lstm_nlayer_seq = lstm_nlayer_seq
        self.dual_rnn_blocks = dual_rnn_blocks

        # Other parameters
        self.dropout_rnn = dropout_rnn
        self.fc_size = fc_size
        self.l2_reg = l2_reg

        # Apply regularization if not None or 0
        self.regularizer = regularizers.l2(self.l2_reg) if self.l2_reg else None

        # Initialize the model components
        self.filterbank_layer = FilterbankLayer(
            nfilter=self.nfilter,
            nfft=self.nfft,
            samplerate=self.samplerate,
            lowfreq=self.lowfreq,
            highfreq=self.highfreq,
            freq_bins=self.freq_bins,
        )

        self.epoch_encoder = BidirectionalRNNLayerBN(
            nhidden=self.nhidden_epoch,
            nlayer=self.lstm_nlayer_epoch,
            seq_len=self.frame_seq_len,
            input_keep_prob=self.dropout_rnn,
            output_keep_prob=self.dropout_rnn,
            regularizer=self.regularizer
        )

        self.attention_layer = AttentionLayer(
            hidden_size=self.nhidden_epoch * 2,
            attention_size=self.attention_size,
            regularizer=self.regularizer
        )

        self.sequence_encoder = SequenceEncoder(
            nhidden_seq=self.nhidden_seq,
            lstm_nlayer_seq=self.lstm_nlayer_seq,
            nsubseq=self.nsubseq,
            sub_seq_len=self.sub_seq_len,
            dual_rnn_blocks=self.dual_rnn_blocks,
            dropout_rnn=self.dropout_rnn,
            regularizer=self.regularizer
        )

        self.fc1 = Dense(
            units=self.fc_size,
            activation='relu',
            kernel_regularizer=self.regularizer)

        self.fc2 = Dense(
            units=self.fc_size,
            activation='relu',
            kernel_regularizer=self.regularizer)

        self.out_layer = Dense(
            units=self.n_classes,
            activation='softmax',
            kernel_regularizer=self.regularizer)

        """ 
        In this case we're using subclassing rather than functional due to custom logic and conditionals, no need to build the model

        if build:
            inputs, outputs = self.init_model()
            super(LSeqSleepNet, self).__init__(inputs=inputs, outputs=outputs)
        """

        # Log the model definition
        if not no_log:
            self.log()

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_classes': self.n_classes,
            'batch_shape': self.batch_shape,
            'nsubseq': self.nsubseq,
            'nfilter': self.nfilter,
            'nfft': self.nfft,
            'samplerate': self.samplerate,
            'lowfreq': self.lowfreq,
            'highfreq': self.highfreq,
            'nhidden_epoch': self.nhidden_epoch,
            'nhidden_seq': self.nhidden_seq,
            'lstm_nlayer_epoch': self.lstm_nlayer_epoch,
            'lstm_nlayer_seq': self.lstm_nlayer_seq,
            'dual_rnn_blocks': self.dual_rnn_blocks,
            'dropout_rnn': self.dropout_rnn,
            'attention_size': self.attention_size,
            'fc_size': self.fc_size,
            'l2_reg': self.l2_reg,
            'no_log': True
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None):
        # throw away the last epoch as the loading will give an odd number of epochs (if needed)
        # X = Lambda(lambda x: x[:, :-1, :, :, :])(inputs)  # (B, L, T, F, C)
        X = tf.reshape(inputs, [-1, self.freq_bins, self.channels])
        # filterbank layer
        X = self.filterbank_layer(X)  # (B*L*T, n_filter*C)

        # epoch encoding
        X = Lambda(lambda x: tf.reshape(x, [-1, self.frame_seq_len, self.nfilter * self.channels]))(
            X)  # (B*L, T, n_filter*C)
        X, _ = self.epoch_encoder(X, training=training)  # (B*L, T, nhidden_epoch*2)
        X = self.attention_layer(X)  # (B*L, nhidden_epoch*2)

        # long sequence modelling with dual encoder
        X = Lambda(lambda x: tf.reshape(x, [-1, self.nsubseq, self.sub_seq_len, self.nhidden_epoch * 2]))(
            X)  # (B, L_sub, K, nhidden_epoch*2)

        X = self.sequence_encoder(X, training=training)  # (B, L_sub, K, nhidden_seq*2)

        # output classifier
        X = Lambda(lambda x: tf.reshape(x, [-1, self.nhidden_seq * 2]))(X)  # (B*L, nhidden_seq*2)
        X = self.fc1(X)  # (B*L, fc_size)
        X = Dropout(rate=1 - self.dropout_rnn)(X, training=training)
        X = self.fc2(X)  # (B*L, fc_size)
        X = Dropout(rate=1 - self.dropout_rnn)(X, training=training)
        out = self.out_layer(X)  # (B*L, n_classes)
        out = Lambda(lambda x: tf.reshape(x, [-1, self.epoch_seq_len, self.n_classes]))(out)  # (B, L, n_classes)

        return out

    def log(self):
        logger.info(f"{self.model_name} Summary\n"
                    "--------------------\n"
                    f"N periods:             {self.epoch_seq_len}\n"
                    f"Time bins:             {self.frame_seq_len}\n"
                    f"Frequency bins:        {self.freq_bins}\n"
                    f"N classes:             {self.n_classes}\n"
                    f"N subseq:              {self.nsubseq}\n"
                    f"N subseq length:       {self.sub_seq_len}\n"
                    f"Filterbank shape:      {self.nfilter}\n"
                    f"N hidden epoch enc:    {self.nhidden_epoch}\n"
                    f"LSTM layers epoch enc: {self.lstm_nlayer_epoch}\n"
                    f"N hidden seq enc:      {self.nhidden_seq}\n"
                    f"LSTM layers seq enc:   {self.lstm_nlayer_seq}\n"
                    f"Attention size:        {self.attention_size}\n"
                    f"L2 regularization:     {self.l2_reg}\n"
                    f"Dropout RNN:           {self.dropout_rnn}\n"
                    f"FC size:               {self.fc_size}\n"
                    f"Dual RNN blocks:       {self.dual_rnn_blocks}\n"
                    "--------------------\n")
