import ops

import tensorflow as tf


def rasor_net(context, context_length, question, question_length, config):
    # Passage-aligned question representations
    context_dense = tf.layers.dense(
        inputs=context,
        units=context.get_shape()[-1].value
    )

    question_dense = tf.layers.dense(
        inputs=question,
        units=question.get_shape()[-1].value,
    )

    s_passage_aligned = tf.matmul(context_dense, tf.transpose(question_dense, perm=[0, 2, 1]))
    attn_passage_aligned = tf.nn.softmax(s_passage_aligned, axis=-1)
    question_aligned = tf.matmul(attn_passage_aligned, question_dense)

    # Passage-independent quesiton representation
    question_lstm_emb, _ = ops.bidirectional_lstm(
        inputs=question,
        input_lengths=question_length,
        size=50,
        name='passage_independent_bLSTM'
    )

    import code
    code.interact(local=locals())

    s_passage_independent = tf.layers.dense(
        inputs=question_lstm_emb,
        units=question.get_shape()[-1].value,
    )

    attn_passage_independent = tf.nn.softmax(s_passage_independent, axis=-1)
    question_independent = tf.matmul(
        attn_passage_independent,
        tf.transpose(s_passage_independent, perm=[0, 2, 1])
    )

    # Concatenation
    
    # Span representations

    import code
    code.interact(local=locals())
