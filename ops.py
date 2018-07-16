import tensorflow as tf
import numpy as np


def bidirectional_lstm(inputs, name, size, is_training=True, input_lengths=None):
    """
    Main bi-LSTM used for passing over word embeddings.
    """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):

        cell_fw = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=1.0)

        input_keep_prob = tf.cond(is_training, true_fn=lambda: tf.constant(0.6), false_fn=lambda: tf.constant(1.0))
        output_keep_prob = tf.cond(is_training, true_fn=lambda: tf.constant(1.0), false_fn=lambda: tf.constant(1.0))
        state_keep_prob = tf.cond(is_training, true_fn=lambda: tf.constant(0.8), false_fn=lambda: tf.constant(1.0))

        cell_fw = tf.contrib.rnn.DropoutWrapper(
            cell=cell_fw,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            state_keep_prob=state_keep_prob,
            variational_recurrent=True,
            input_size=inputs.get_shape()[-1].value,
            dtype=tf.float32
        )

        cell_bw = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=1.0)
        cell_bw = tf.contrib.rnn.DropoutWrapper(
            cell=cell_bw,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            state_keep_prob=state_keep_prob,
            variational_recurrent=True,
            input_size=inputs.get_shape()[-1].value,
            dtype=tf.float32
        )

        outputs_tuple, final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            dtype=tf.float32,
            sequence_length=input_lengths,
            inputs=inputs
        )

        outputs_concat = tf.concat([outputs_tuple[0], outputs_tuple[1]], axis=-1)

    return outputs_concat, final_states


def embed_sequence(inputs, name, embedding_matrix):
    """
    Used to embed sequences of word ids.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding_matrix = tf.get_variable(
            name="embedding_matrix",
            initializer=tf.constant(embedding_matrix,
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


def normalize_linear(embedded_sequence):
    """ embedded_sequence: [batch size x example length x embeding size] """
    row_sums = tf.reduce_sum(embedded_sequence, axis=-1)
    row_sums = tf.expand_dims(row_sums, 1) + 1e-32  # prevent NaNs

    # Normalize
    normalized = tf.transpose(tf.divide(tf.transpose(embedded_sequence, perm=[0, 2, 1]), row_sums), perm=[0, 2, 1])

    return normalized


def mask(sequence_lengths, sequence):
    with tf.name_scope('mask', values=[sequence_lengths, sequence]):
        max_len = sequence.get_shape()[1].value
        mask = tf.sequence_mask(sequence_lengths, max_len, dtype=tf.float32)
        masked_seq = sequence * tf.expand_dims(mask, -1)
        return masked_seq


def sequence_softmax(sequence_weights, sequence_length, scope='sequence_softmax'):
    with tf.name_scope(scope, values=[sequence_weights, sequence_length]):
        max_sequenence_length = sequence_weights.get_shape()[1].value
        mask_bs = tf.sequence_mask(
            sequence_length,
            maxlen=max_sequenence_length,
            dtype=tf.float32,
            name='pad_weights_removal_mask'
        )  # b x sA
        weights_a_exp_masked = mask_bs * tf.exp(sequence_weights)  # b x s (all exponentiated, but padded locs remain 0)
        weights_norm = l1_normalize(weights_a_exp_masked)
    return weights_norm


def l1_normalize(unnormalized_values):
    with tf.name_scope('l1_normalize', values=[unnormalized_values]):
        normalized = unnormalized_values / tf.reduce_sum(unnormalized_values, axis=-1, keep_dims=True)
    return normalized


def masked_softmax(seq, mask):
    exp_seq = tf.exp(seq)
    masked_exp_seq = mask * exp_seq
    _masked_softmax = mask * normalize_linear(masked_exp_seq)
    return _masked_softmax

