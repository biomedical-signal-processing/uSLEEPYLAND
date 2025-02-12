"""
Implementation of SleepTransformer model, as described in the paper:

Phan, Huy, et al. Sleeptransformer: Automatic sleep staging with
interpretability and uncertainty quantification.
IEEE Transactions on Biomedical Engineering 69.8 (2022): 2456-2467.
"""

import logging
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Layer, Dense, LayerNormalization,
                                     Dropout, Embedding, Lambda, Input
                                     )
import numpy as np

logger = logging.getLogger(__name__)

@tf.keras.saving.register_keras_serializable(package='Custom', name='MultiHeadAttention')
class MultiHeadAttention(Layer):

    def __init__(self,
                 d_model,
                 num_heads,
                 regularizer=None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model  # Frequency bins * channels
        self.regularizer = regularizer

        self.depth = d_model // num_heads  # d_q, d_k, d_v

        self.wq = Dense(units=d_model,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer)
        self.wk = Dense(units=d_model,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer)
        self.wv = Dense(units=d_model,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer)

        self.dense = Dense(units=d_model,
                           kernel_regularizer=regularizer,
                           bias_regularizer=regularizer)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['regularizer'] = tf.keras.regularizers.deserialize(config['regularizer'])
        return cls(**config)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (B, seq_len, d_model)
        k = self.wk(k)  # (B, seq_len, d_model)
        v = self.wv(v)  # (B, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (B, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (B, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (B, num_heads, seq_len_v, depth)

        scaled_attention = self.scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (B, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (B, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (B, seq_len_q, d_model)

        return output

    def scaled_dot_product_attention(self, q, k, v):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (B, num_heads, seq_len_q, seq_len_k)

        # Scale matmul_qk by dividing the square root of dk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Clip the logits to prevent extreme values that could lead to numerical instability
        scaled_attention_logits = tf.clip_by_value(scaled_attention_logits, -10.0, 10.0)

        # Softmax on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B, num_heads, seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (B, num_heads, seq_len_q, depth_v)

        return output


@tf.keras.saving.register_keras_serializable(package='Custom', name='EncoderLayer')
class EncoderLayer(Layer):

    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 attn_dropout_rate=0.1,
                 ff_dropout_rate=0.1,
                 activation_fn='relu',
                 ln_epsilon=1e-8,
                 regularizer=None):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.attn_dropout_rate = attn_dropout_rate
        self.ff_dropout_rate = ff_dropout_rate
        self.ln_epsilon = ln_epsilon
        self.regularizer = regularizer
        self.activation_fn = activation_fn

        self.dropout1 = Dropout(rate=attn_dropout_rate)
        self.dropout2 = Dropout(rate=ff_dropout_rate)
        self.layernorm = LayerNormalization(epsilon=ln_epsilon)
        self.activation_fn = activation_fn

        self.mha = MultiHeadAttention(d_model=d_model,
                                      num_heads=num_heads,
                                      regularizer=regularizer)
        self.ffn = self.point_wise_feed_forward_network(d_model=d_model,
                                                        dff=dff,
                                                        regularizer=regularizer)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'attn_dropout_rate': self.attn_dropout_rate,
            'ff_dropout_rate': self.ff_dropout_rate,
            'activation_fn': self.activation_fn,
            'ln_epsilon': self.ln_epsilon,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['regularizer'] = tf.keras.regularizers.deserialize(config['regularizer'])
        return cls(**config)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)  # (B, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection and layer normalization
        out1 = self.layernorm(x + attn_output)  # (B, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (B, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection and layer normalization
        out2 = self.layernorm(out1 + ffn_output)

        return out2

    def point_wise_feed_forward_network(self,
                                        d_model,
                                        dff,
                                        regularizer=None):
        return tf.keras.Sequential([
            Dense(dff,
                  activation=self.activation_fn,
                  kernel_regularizer=regularizer,
                  bias_regularizer=regularizer),  # (B, seq_len, dff)
            Dense(d_model,
                  kernel_regularizer=regularizer,
                  bias_regularizer=regularizer)  # (B, seq_len, d_model)
        ])


@tf.keras.saving.register_keras_serializable(package='Custom', name='Encoder')
class Encoder(Layer):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 maximum_position_encoding,
                 attn_dropout_rate=0.1,
                 ff_dropout_rate=0.1,
                 activation_fn='relu',
                 ln_epsilon=1e-8,
                 regularizer=None):
        super(Encoder, self).__init__()

        self.d_model = d_model  # Frequency bins * channels
        self.num_layers = num_layers  # Number of encoder layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.attn_dropout_rate = attn_dropout_rate
        self.ff_dropout_rate = ff_dropout_rate
        self.activation_fn = activation_fn
        self.ln_epsilon = ln_epsilon
        self.regularizer = regularizer

        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        attn_dropout_rate=attn_dropout_rate,
                                        ff_dropout_rate=ff_dropout_rate,
                                        activation_fn=activation_fn,
                                        ln_epsilon=ln_epsilon,
                                        regularizer=regularizer) for _ in range(num_layers)]

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'attn_dropout_rate': self.attn_dropout_rate,
            'ff_dropout_rate': self.ff_dropout_rate,
            'activation_fn': self.activation_fn,
            'ln_epsilon': self.ln_epsilon,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['regularizer'] = tf.keras.regularizers.deserialize(config['regularizer'])
        return cls(**config)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]  # T, in practice equal to maximum_position_encoding

        # Embedding isn't necessary here
        # x (B*L, T, F*C), each time step is a vector of F*C dimensions
        # we simply scale and add the positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # (B*L, T, F*C)
        x += self.pos_encoding[:, :seq_len, :]  # (B*L, T, F*C)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)  # (B*L, T, F*C)

        return x  # (B*L, T, F*C)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


@tf.keras.saving.register_keras_serializable(package='Custom', name='AttentionLayer')
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
                                       regularizer=regularizer,
                                       name='W_omega')
        self.b_omega = self.add_weight(shape=(attention_size,),
                                       initializer=tf.random_normal_initializer(stddev=0.1),
                                       regularizer=regularizer,
                                       name='b_omega')
        self.u_omega = self.add_weight(shape=(attention_size,),
                                       initializer=tf.random_normal_initializer(stddev=0.1),
                                       regularizer=regularizer,
                                       name='u_omega')

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'attention_size': self.attention_size,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['regularizer'] = tf.keras.regularizers.deserialize(config['regularizer'])
        return cls(**config)

    def call(self, inputs):
        # inputs have shape (B*L, T, F*C) where F*C is the hidden_size
        # this is essentially just a dense layer with tanh
        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, self.hidden_size]),
                              self.W_omega) + self.b_omega)  # (B*L*T, attention_size)

        # multiply by epoch-level context vector
        vu = tf.matmul(v, tf.reshape(self.u_omega, [-1, 1]))  # (B*L*T, 1)

        # Clip the values of vu to prevent extreme values
        vu = tf.clip_by_value(vu, -10.0, 10.0)

        # Reshape to [batch_size, sequence_length] and apply softmax to get the attention weights
        exps = tf.reshape(tf.exp(vu), [-1, inputs.shape[1]])  # (B*L, T)
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  # (B*L, T)

        # Compute the epoch representation as a weighted sum
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, inputs.shape[1], 1]), 1)  # (B*L, F*C)

        return output


@tf.keras.saving.register_keras_serializable(package='Custom', name='SleepTransformer')
class SleepTransformer(Model):

    def __init__(self,
                 n_classes,
                 batch_shape,
                 frm_d_model,
                 frm_num_blocks,
                 frm_num_heads,
                 frm_dff,
                 seq_d_model,
                 seq_num_blocks,
                 seq_num_heads,
                 seq_dff,
                 attention_frame_size,
                 fc_hidden_size,
                 attn_dropout_rate,
                 ff_dropout_rate,
                 fc_dropout_rate,
                 ln_epsilon,
                 ff_activation_fn,
                 l2_reg=None,
                 build=True,
                 no_log=False,
                 **kwargs
                 ):
        """
        Args:
            n_classes (int):
                Number of classes to predict
            batch_shape (list):
                Shape of the input batch [B, L, T, F, C]
            frm_d_model (int):
                Internal dimension of the epoch transformer
            frm_num_blocks (int):
                Number of encoder blocks in the epoch transformer
            frm_num_heads (int):
                Number of mha in the epoch transformer
            frm_dff (int):
                Hidden layer size in the epoch transformer for FFN part
            seq_d_model (int):
                Internal dimension of the sequence transformer
            seq_num_blocks (int):
                Number of encoder blocks in the sequence transformer
            seq_num_heads (int):
                Number of mha in the sequence transformer
            seq_dff (int):
                Hidden layer size in the sequence transformer for FFN part
            attention_frame_size (int):
                Dimensionality of attention layer in between the two transformers
            fc_hidden_size (int):
                Hidden layer size in the final output classifier
            attn_dropout_rate (float):
                Dropout rate for the attention layers
            ff_dropout_rate (float):
                Dropout rate for the feed forward layers
            fc_dropout_rate (float):
                Dropout rate for the final output classifier
            ln_epsilon (float):
                Epsilon for layer normalization, by default 1e-8
            ff_activation_fn (str):
                Activation function for the feed forward layers, by default 'relu'
            l2_reg (float or None):
                L2 regularization parameter, by default None
            build (bool):
                Whether to build the model, by default True
            no_log (bool):
                Whether to avoid logging the model, by default False
        """

        self.model_name = "SleepTransformer"

        # input and output shapes
        assert len(batch_shape) == 5  # (B, L, T, F, C)
        self.n_classes = n_classes
        self.batch_shape = batch_shape
        self.epoch_seq_len = batch_shape[1]  # L
        self.frame_seq_len = batch_shape[2]  # T
        self.freq_bins = batch_shape[3] - 1  # F
        self.channels = batch_shape[4]  # C

        # transformer parameters
        # Epoch transformer (frm) parameters
        self.frm_d_model = frm_d_model
        self.frm_num_blocks = frm_num_blocks
        self.frm_num_heads = frm_num_heads
        self.frm_dff = frm_dff
        self.frm_maxlen = self.frame_seq_len
        # Sequence transformer (seq) parameters
        self.seq_d_model = seq_d_model
        self.seq_num_blocks = seq_num_blocks
        self.seq_num_heads = seq_num_heads
        self.seq_dff = seq_dff
        self.seq_maxlen = self.epoch_seq_len
        # Epoch representation parameters
        self.attention_frame_size = attention_frame_size
        # output layer parameters
        self.fc_hidden_size = fc_hidden_size
        self.fc_dropout_rate = fc_dropout_rate
        # parameters for both transformers
        self.attn_dropout_rate = attn_dropout_rate
        self.ff_dropout_rate = ff_dropout_rate
        self.ln_epsilon = ln_epsilon
        self.ff_activation_fn = ff_activation_fn

        # L2 regularization
        self.l2_reg = l2_reg

        # Build model and init base keras Model class
        if build:
            inputs, outputs = self.init_model()
            super(SleepTransformer, self).__init__(inputs=inputs, outputs=outputs)

        # Log the model definition
        if not no_log:
            self.log()

    def get_config(self):
        config = super(SleepTransformer, self).get_config()
        config.update({
            'n_classes': self.n_classes,
            'batch_shape': self.batch_shape,
            'frm_d_model': self.frm_d_model,
            'frm_num_blocks': self.frm_num_blocks,
            'frm_num_heads': self.frm_num_heads,
            'frm_dff': self.frm_dff,
            'seq_d_model': self.seq_d_model,
            'seq_num_blocks': self.seq_num_blocks,
            'seq_num_heads': self.seq_num_heads,
            'seq_dff': self.seq_dff,
            'attention_frame_size': self.attention_frame_size,
            'fc_hidden_size': self.fc_hidden_size,
            'attn_dropout_rate': self.attn_dropout_rate,
            'ff_dropout_rate': self.ff_dropout_rate,
            'fc_dropout_rate': self.fc_dropout_rate,
            'ln_epsilon': self.ln_epsilon,
            'ff_activation_fn': self.ff_activation_fn,
            'l2_reg': self.l2_reg,
            'build': True,  # note that this is necessary for model loading
            'no_log': True,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _check_for_extremes(self, tensor, name):
        nan_check = tf.reduce_any(tf.math.is_nan(tensor))
        inf_check = tf.reduce_any(tf.math.is_inf(tensor))
        too_large_check = tf.reduce_any(tf.abs(tensor) > 1e8)

        if nan_check:
            logger.info(f"{name} - Contains NaN:", nan_check)
        if inf_check:
            logger.info(f"{name} - Contains Inf:", inf_check)
        if too_large_check:
            logger.info(f"{name} - Contains extremely large values:", too_large_check)

        return tensor

    def _output_module(self, inputs, regularizer):
        # final output classifier on top with three dense layers and dropout
        z = Dense(self.fc_hidden_size,
                  activation=self.ff_activation_fn,
                  kernel_regularizer=regularizer,
                  bias_regularizer=regularizer)(inputs)
        z = Dropout(self.fc_dropout_rate)(z)
        z = Dense(self.fc_hidden_size,
                  activation=self.ff_activation_fn,
                  kernel_regularizer=regularizer,
                  bias_regularizer=regularizer)(z)
        z = Dropout(self.fc_dropout_rate)(z)
        # separate linear layer and softmax for numerical stability
        out = Dense(self.n_classes,
                    activation=None,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(z)
        # Clip the logits before applying softmax to avoid extreme values (uncomment if from_logits = False)
        out = tf.clip_by_value(out, -10.0, 10.0)
        # out = tf.nn.softmax(out)
        # reshape to expected output shape (B, L, n_classes)
        out = Lambda(lambda x: tf.reshape(x, [-1, self.epoch_seq_len, self.n_classes]))(out)

        return out

    def init_model(self):

        # Apply regularization if not None or 0
        regularizer = regularizers.l2(self.l2_reg) if self.l2_reg else None

        epoch_transformer = Encoder(num_layers=self.frm_num_blocks,
                                    d_model=self.frm_d_model,
                                    num_heads=self.frm_num_heads,
                                    dff=self.frm_dff,
                                    maximum_position_encoding=self.frm_maxlen,
                                    attn_dropout_rate=self.attn_dropout_rate,
                                    ff_dropout_rate=self.ff_dropout_rate,
                                    activation_fn=self.ff_activation_fn,
                                    ln_epsilon=self.ln_epsilon,
                                    regularizer=regularizer)

        attn_layer = AttentionLayer(self.frm_d_model,
                                    self.attention_frame_size,
                                    regularizer=regularizer)

        sequence_transformer = Encoder(num_layers=self.seq_num_blocks,
                                       d_model=self.seq_d_model,
                                       num_heads=self.seq_num_heads,
                                       dff=self.seq_dff,
                                       maximum_position_encoding=self.seq_maxlen,
                                       attn_dropout_rate=self.attn_dropout_rate,
                                       ff_dropout_rate=self.ff_dropout_rate,
                                       activation_fn=self.ff_activation_fn,
                                       ln_epsilon=self.ln_epsilon,
                                       regularizer=regularizer)

        # inputs shape (B, L, T, F, C)
        inputs = Input(shape=[
            self.epoch_seq_len,
            self.frame_seq_len,
            self.freq_bins + 1,
            self.channels],
            name="input")

        # eliminate the 0-th frequency bin as in the paper
        z = Lambda(lambda x: x[:, :, :, 1:, :])(inputs)

        # reshape input to (B*L, T, F*C)
        z = Lambda(lambda x: tf.reshape(x, [-1, self.frame_seq_len, self.freq_bins * self.channels]))(z)

        z = epoch_transformer(z)  # (B*L, T, F*C)

        z = attn_layer(z)  # (B*L, F*C)

        # reshape to (B, L, F*C) to process the sequent of adjacent epochs
        z = Lambda(lambda x: tf.reshape(x, [-1, self.epoch_seq_len, self.frm_d_model]))(z)
        z = sequence_transformer(z)  # (B, L, F*C)

        # reshape to (B*L, F*C)
        z = Lambda(lambda x: tf.reshape(x, [-1, self.frm_d_model]))(z)

        # final output classifier
        out = self._output_module(z, regularizer)

        return [inputs], [out]

    def log(self):
        logger.info(f"{self.model_name} Summary\n"
                    "--------------------\n"
                    f"N periods:         {self.epoch_seq_len}\n"
                    f"Time bins:         {self.frame_seq_len}\n"
                    f"Frequency bins:    {self.freq_bins}\n"
                    f"N classes:         {self.n_classes}\n"
                    f"d model:           {self.frm_d_model}\n"
                    f"Encoders:          {self.frm_num_blocks}\n"
                    f"Heads:             {self.frm_num_heads}\n"
                    f"ff dim:            {self.frm_dff}\n"
                    f"d model seq:       {self.seq_d_model}\n"
                    f"Encoders seq:      {self.seq_num_blocks}\n"
                    f"Heads seq:         {self.seq_num_heads}\n"
                    f"ff dim seq:        {self.seq_dff}\n"
                    f"ff activation:     {self.ff_activation_fn}\n"
                    f"Attention frame:   {self.attention_frame_size}\n"
                    f"FC hidden units:   {self.fc_hidden_size}\n"
                    f"l2 reg:            {self.l2_reg}\n"
                    
                    f"N params:          {self.count_params()}\n"
                    f"Input:             {self.input}\n"
                    f"Output:            {self.output}"
                    )
