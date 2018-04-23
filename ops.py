import tensorflow as tf
import numpy as np


def bidirectional_lstm(inputs, name, size, input_lengths=None):
    """
    Main bi-LSTM used for passing over word embeddings.
    """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        cell_fw = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=1.0)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=1.0)

        outputs_tuple, final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            dtype=tf.float32,
            sequence_length=input_lengths,
            inputs=inputs
        )

        outputs_concat = tf.concat([outputs_tuple[0], outputs_tuple[1]], axis=-1)

    return outputs_concat, final_states


def embed_sequence(inputs, name):
    """
    Used to embed sequences of word ids.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding_matrix = tf.get_variable(
            name="embedding_matrix",
            initializer=tf.constant(np.load(open('data/embedding_matrix.npy', 'rb')),
                                    dtype=tf.float32,
                                    name="embedding_matrix_init"),
            trainable=False)

        embs = tf.nn.embedding_lookup(params=embedding_matrix,
                                      ids=inputs)

    return embs


def unpadded_lengths(tensor):
    """ input: batch size x input size x emb size """
    used = tf.sign(tensor)
    lengths = tf.reduce_sum(used, axis=-1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths

