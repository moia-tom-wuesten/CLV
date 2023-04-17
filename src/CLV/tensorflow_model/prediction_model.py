# %%
import tensorflow as tf
import os.path

import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Lambda
import keras.backend as K


class PredictionModel:

    """
    Creates a class with several functions for training model preparations
    and fit as instances.

    :params: values for design of Tensorflow graph (e. g. time_units, max_features),
    list of features, number of neurons for dense/lstm layers

    :return: class with training model setup and fit
    """

    def __init__(
        self, prediction_batch_size: int, model_weights_filename: str, training_model
    ):
        self.prediction_batch_size = prediction_batch_size
        self.model_weights_filename = model_weights_filename
        self.training_model = training_model

    def graph(self):
        # the prediction model is almost identical to the training model
        # the one difference is that instead of a probability distribution
        # output, we use a sampling layer to create realizations to simulate
        # alternative future scenarios
        # number of units in the memory layer
        memory_units = 128

        def sample_multinomial(probs):
            """
            Draws a sample of length 1 from a multinomial distribution with the
            given class probabilities.
            """

            return tf.cast(
                tf.expand_dims(
                    tfp.distributions.Categorical(probs=probs).sample(), axis=-1
                ),
                dtype=K.floatx(),
            )

        # sample values from softmax multinomial distribution
        sample_layer = Lambda(sample_multinomial, name="sample_transactions")

        """ Another difference is that we use a "stateful" LSTM layer for prediction.
            This layer has the exact same size as the LSTM we used in the training model,
            hence the same number of parameters, and we will literally copy the parameters
            over from the training model. The "stateful" property means that the internal
            cell state - the "memories" of the layer - are kept until we explicitly delete
            them. In trainin those are reset to 0 after each sequence is processed - we're
            learning from independent histories, so keeping the memories doesn't make sense
            there. In prediction however, we need a bit of fine control: we will be feeding
            new inputs into the model step by step, so we will also be carefully managing the
            memory content ourselves.
        """

        prediction_memory_layer = LSTM(
            memory_units, return_sequences=True, stateful=True, name="lstm"
        )

        # we use a separate set of Input objects for prediction since the shape of
        # data is different:
        # the first dimension is the BATCH_SIZE_PRED
        # the second dimension is NONE because instead of knowing the length of the
        # sequence beforehand we want the model to accept sequences of arbitrary length
        # the last dimension is 1 (1 scalar value per timestep per feature)

        p_input_week = Input(
            batch_shape=(self.prediction_batch_size, None, 1), name="week"
        )
        p_input_transactions = Input(
            batch_shape=(self.prediction_batch_size, None, 1), name="transaction"
        )

        # reuse the pretrained embeddings
        emb_week = self.training_model.embedding_week(p_input_week)
        emb_trans = self.training_model.embedding_transactions(p_input_transactions)

        # squeeze out a superfluous dimension from the tensor like before
        emb_week = self.training_model.squeeze(emb_week)

        emb_trans = self.training_model.squeeze(emb_trans)

        # combine embedded vectors into a single vector
        output = self.training_model.concat([emb_week, emb_trans])

        # apply the prediction memory layer
        output = prediction_memory_layer(output)

        # feed-forward through the dense layers
        output = self.training_model.dense_layer(output)

        # softmax layer
        output = self.training_model.softmax_layer(output)

        # apply final sampling layer
        output = sample_layer(output)

        # build the prediction model
        prediction_inputs = [p_input_week, p_input_transactions]
        model_pred = Model(prediction_inputs, output)

        # the training and prediction models have the exact same number of parameters
        # assert model_pred.count_params() == model_train.count_params()
        if self.check_training_model():
            # check if model has already been trained
            model_pred.load_weights(self.model_weights_filename, by_name=True)
        return model_pred

    def check_training_model(self):
        if not os.path.exists(self.model_weights_filename) and os.path.isfile(
            self.model_weights_filename
        ):
            print("please go back and train a model first")
            return False
        else:
            return True
