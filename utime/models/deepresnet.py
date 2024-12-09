import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Bidirectional,
                                     Conv1D, Conv2D, GRU, Input, Lambda,
                                     AveragePooling1D, MaxPooling2D
                                     )

logger = logging.getLogger(__name__)

class DeepResNet(Model):
    def __init__(self,
                 batch_shape,
                 n_classes,
                 filter_base,
                 kernel_size,
                 max_pooling,
                 num_blocks,
                 rnn_bidirectional,
                 rnn_num_layers,
                 rnn_num_units,
                 sec_per_prediction=None,
                 no_log=False,
                 name="",
                 **unused):

        # Set various attributes
        assert len(batch_shape) == 4
        self.n_periods = batch_shape[1]
        self.input_dims = batch_shape[2]
        self.n_channels = batch_shape[3]
        self.n_classes = int(n_classes)
        self.filter_base = filter_base
        self.kernel_size = int(kernel_size)
        self.max_pooling = max_pooling
        self.num_blocks = num_blocks
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_num_layers = rnn_num_layers
        self.rnn_num_units = rnn_num_units if rnn_num_units is not None \
            else 4 * self.filter_base * (2 ** (self.num_blocks - 1))

        self.sec_per_prediction = sec_per_prediction
        if not isinstance(self.sec_per_prediction, (int, np.integer)):
            raise TypeError("data_per_prediction must be an integer value")

        self.model_name = "DeepResNet"

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Log the model definition
        if not no_log:
            self.log()

    def _mixing_module(self, inputs):
        z = Conv2D(self.n_channels, (self.n_channels, 1))(inputs)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        return z

    def _build_encoder(self,
                       inputs):
        z = inputs
        block = None
        for k in range(self.num_blocks):
            # Define shortcut
            shortcut = (Conv2D(filters=4 * self.filter_base * (2 ** k), kernel_size=(1, 1))
                        (z if k == 0 else block))

            # Define basic block structure
            block = (Conv2D(filters=self.filter_base * (2 ** k), kernel_size=(1, 1))
                     (z if k == 0 else block))
            block = BatchNormalization()(block)
            block = Activation('relu')(block)
            block = Conv2D(filters=self.filter_base * (2 ** k), kernel_size=(1, self.kernel_size),
                           padding='same')(block)
            block = BatchNormalization()(block)
            block = Activation('relu')(block)
            block = Conv2D(filters=4 * self.filter_base * (2 ** k), kernel_size=(1, 1))(block)
            block = BatchNormalization()(block)
            block += shortcut
            block = Activation('relu')(block)
            block = MaxPooling2D(pool_size=(1, self.max_pooling))(block)

        return block

    def init_model(self, inputs=None):
        """
        Build the ResNet model with the specified input shape.
        """
        if inputs is None:
            inputs = Input(shape=[self.n_periods,
                                  self.input_dims,
                                  self.n_channels],
                           name="input")

        reshaped_1 = [-1, self.n_channels, self.n_periods * self.input_dims, 1]
        z = Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(inputs)
        z = Lambda(lambda x: tf.reshape(x, reshaped_1))(z)

        if self.n_channels != 1:
            z = self._mixing_module(z)

        enc = self._build_encoder(z)

        enc = Lambda(lambda x: tf.squeeze(x, axis=1))(enc)

        # TODO - self.rnn_num_layers == 1
        temp_rnn = Bidirectional(GRU(units=self.rnn_num_units,
                                     return_sequences=True,
                                     dropout=0,
                                     recurrent_dropout=0),
                                 merge_mode='concat')(enc)

        out = Conv1D(filters=self.n_classes,
                     kernel_size=1)(temp_rnn)
        out = AveragePooling1D((self.sec_per_prediction))(out)
        out = Activation('softmax')(out)

        return [inputs], [out]

    def log(self):
        logger.info(f"\nDeepResNet Model Summary\n"
                    "--------------------\n"
                    f"N periods:            {self.n_periods or 'ANY'}\n"
                    f"Input dims:           {self.input_dims}\n"
                    f"N channels:           {self.n_channels}\n"
                    f"N classes:            {self.n_classes}\n"
                    f"Kernel size:          {self.kernel_size}\n"
                    f"Filter base:          {self.filter_base}\n"
                    f"Max pooling:          {self.max_pooling}\n"
                    f"Num blocks:           {self.num_blocks}\n"
                    f"RNN bidirectional:    {self.rnn_bidirectional}\n"
                    f"RNN num layers:       {self.rnn_num_layers}\n"
                    f"RNN num units:        {self.rnn_num_units}\n"
                    f"Seq length.:          {self.n_periods * self.input_dims}\n"
                    f"N params:             {self.count_params()}\n"
                    f"Input:                {self.input}\n"
                    f"Output:               {self.output}")