# %%
from typing import Optional

# %%
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import sparse_categorical_crossentropy
import keras.backend as K
from keras.utils.vis_utils import plot_model


# %%
class TrainingModel:

    """
    Creates a class with several functions for training model preparations
    and fit as instances.

    :params: values for design of Tensorflow graph (e. g. time_units, max_features),
    list of features, number of neurons for dense/lstm layers

    :return: class with training model setup and fit
    """

    def __init__(self, max_weeks, max_trans, seq_len, max_epoch, name):
        self.max_weeks = max_weeks
        self.max_trans = max_trans
        self.seq_len = seq_len
        self.max_epoch = max_epoch
        self.name = name

    def graph(self):

        """
        Creates the model setup as a Tensorflow graph.

        :params: instances from class TrainingModel, weeks_dims as number of
        output dimensions for week layer

        :return: setup Keras model graph
        """

        # number of units in the memory layer
        memory_units = 128

        # number of units in the dense layer
        dense_units = 128

        def emb_size(feature_max: int):
            """A simple heuristic to determine embedding layer size"""
            return int(feature_max**0.5) + 1

        """ input layers map directly to the features from the input data: week, transactions """
        input_week = Input(shape=(self.seq_len, 1), name="week")
        input_transactions = Input(shape=(self.seq_len, 1), name="transaction")

        """ embedding layers compress signal into a dense real-valued vector representation. This helps the model extract useful signals from the input features """
        self.embedding_week = Embedding(
            self.max_weeks, emb_size(self.max_weeks), name="embed_week"
        )
        self.embedding_transactions = Embedding(
            self.max_trans, emb_size(self.max_trans), name="embed_trans"
        )

        # a simple layer that concatenates a list of vectors
        self.concat = Concatenate(axis=-1, name="concat")

        # a simple layer that removes a dimension from data tensor
        #  this is needed because the embedding layers introduce
        #  an extra dimension (of size 1), which we do not need
        self.squeeze = Lambda(lambda x: K.squeeze(x, axis=-2), name="squeeze")

        """ the LSTM serves as the 'memory' of the model
            we define separate LSTM layers for training and prediction models
            'stateful' LSTM keeps the internal state (memory) until explicitly deleted
            'state-less' LSTM forgets everything after each batch of samples is processed """
        training_memory_layer = LSTM(
            memory_units, return_sequences=True, stateful=False, name="lstm"
        )

        """ Dense layers add non-linear compute capacity to the model """
        self.dense_layer = Dense(
            dense_units, name="dense"
        )  # activation=dense_activation, )

        """ The final output layer is a softmax prediction layer where
            each neuron represents the probability predicting a given transaction count """
        self.softmax_layer = Dense(self.max_trans, activation="softmax", name="softmax")
        # now we assemble the layers into a complete Model
        # first step is to connect the embedding layers with the inputs
        emb_week = self.embedding_week(input_week)
        emb_trans = self.embedding_transactions(input_transactions)

        # we also squeeze out a superfluous dimension from the tensor
        emb_week = self.squeeze(emb_week)
        emb_trans = self.squeeze(emb_trans)

        # now we combine embedded vectors into a single long vector
        output = self.concat([emb_week, emb_trans])

        # pass the result to the memory layer(s)
        output = training_memory_layer(output)

        # feed-forward through the dense layer(s)
        output = self.dense_layer(output)

        # final softmax layer
        output = self.softmax_layer(output)

        # build the training model
        training_inputs = [input_week, input_transactions]
        self.model_train = Model(training_inputs, output)

        return self.model_train

    def model_summary(self, display: Optional[str] = "table"):

        """
        Outputs model graph summaries.

        :params: instances from class TrainingModel, display as "table" or "graph"

        :return: table or graph of Keras model graph
        """

        if display == "table":
            return self.model_train.summary()
        else:
            graph = plot_model(
                self.model_train,
                rankdir="LR",
                to_file="../../plots/model_plot.png",
                show_shapes=True,
                show_layer_names=True,
            )
            return graph

    def train_model(
        self,
        train_dataset,
        valid_dataset,
        no_train_samples,
        no_valid_samples,
        batch_size_train,
        batch_size_val,
    ):

        """
        Trains the specified Keras model.

        :params: instances from class TrainingModel, dataset with independent
        variables (x, x_valid), dependent variable (y, y_valid),
        number of steps, min_delta

        :return: trained model
        """
        self.model_train.reset_states()

        # use the popular Adam optimizer with all default parameters
        optimizer = Adam()

        # compile the model to get it ready for training
        self.model_train.compile(
            loss=sparse_categorical_crossentropy, optimizer=optimizer
        )
        self.model_date_time = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M")
        self.model_weights_filename = (
            f"tensorflow_model/weights/{self.name}_{self.model_date_time}_weights.hdf5"
        )

        callbacks = [
            # monitor the validation loss and stop training after
            # 'patience' number of epochs during which there is
            # no improvement whatsoever, then restore model weights
            # from the very best epoch afterwards (epoch with the
            # lowest validation loss
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                restore_best_weights=True,
                # training will stop early if validation loss stops improving for 5 epochs
                patience=5,
                verbose=1,
                mode="auto",
            ),
            # save model parameters to file as long as the model
            # improves (val_loss decreases)
            ModelCheckpoint(
                self.model_weights_filename,
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ]

        # start the actual training
        # the final validation loss should be around 0.4 with the default hyperparameters
        self.history = self.model_train.fit(
            train_dataset,
            epochs=self.max_epoch,
            verbose=2,
            shuffle=True,
            callbacks=callbacks,
            validation_data=valid_dataset,
            validation_steps=no_valid_samples // batch_size_val,
            steps_per_epoch=no_train_samples // batch_size_train,
        )
