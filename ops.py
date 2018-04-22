import tensorflow as tf


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
