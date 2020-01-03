# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential

import pdb

class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        
        self._num_layers = num_layers
        self._dropout = dropout
        self._layers = []
        self._input_dim = input_dim

        for i in range(num_layers):
            self._layers.append(layers.Dense(input_dim, activation='tanh'))

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        
        mask = tf.expand_dims(tf.cast(sequence_mask, dtype=tf.float32), axis=len(sequence_mask.shape))
        vector_sequence = vector_sequence * mask
        mask2 = None
        if training:
            samples = tf.random.uniform(sequence_mask.shape)
            mask2 = tf.greater(samples, self._dropout)
            mask2 = tf.expand_dims(tf.cast(mask2, dtype=tf.float32), axis=len(sequence_mask.shape))
            vector_sequence = vector_sequence * mask2
        
        # If this is training, mask2 (dropout) will be None
        if mask2 is None:
            mask2 = mask

        # Get the final mask
        mask = mask * mask2
        # Get the number of non-zero tokens 
        nz = tf.math.count_nonzero(mask, axis=1, dtype=tf.dtypes.float32)
        avg_vector_sequence = tf.reduce_sum(vector_sequence, axis = 1)
        # Get average vectors
        avg_vector_sequence = tf.math.divide_no_nan(avg_vector_sequence, nz)

        prev_output = avg_vector_sequence

        layer_representations = []
        for layer in self._layers:
            prev_output = layer(prev_output)
            layer_representations.append(prev_output)
        return {"combined_vector": prev_output,
                "layer_representations": tf.stack(layer_representations, axis=1)}

class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        self._layers = [] 
        self._num_layers = num_layers
        self._input_dim = input_dim
        for layer in range(num_layers):
            self._layers.append(layers.GRU(input_dim, activation='tanh', return_sequences = True, return_state = True))

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:


        prev_output = vector_sequence
        layer_representations = []
        state = None
        sequence = None
        for layer in self._layers:
            (sequence, state) = layer(prev_output, mask = sequence_mask)
            prev_output = sequence
            layer_representations.append(state)
        return {"combined_vector": layer_representations[self._num_layers - 1],
                "layer_representations": tf.stack(layer_representations, axis=1)}