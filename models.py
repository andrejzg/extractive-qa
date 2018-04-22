import ops

import tensorflow as tf


def rasor_net(context, context_length, question, question_length, span2position, config):
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

    # Passage-independent question representation
    question_lstm_emb, _ = ops.bidirectional_lstm(
        inputs=question,
        input_lengths=question_length,
        size=25,
        name='passage_independent_bLSTM'
    )

    s_passage_independent = tf.layers.dense(
        inputs=question_lstm_emb,
        units=1,
    )

    attn_passage_independent = tf.nn.softmax(tf.squeeze(s_passage_independent, axis=-1), axis=-1)
    question_independent = attn_passage_independent * tf.transpose(question_lstm_emb, perm=[0, 2, 1])
    question_independent = tf.transpose(question_independent, perm=[0, 2, 1])

    question_independent_final = tf.reduce_sum(question_independent, axis=1)
    question_independent_final_stacked = tf.stack([question_independent_final] * context.get_shape()[1].value, axis=1)

    # Concatenation
    context_query_aware = tf.concat([question_aligned, context, question_independent_final_stacked], axis=-1)

    # Span representations
    context_query_aware_lstm, _ = ops.bidirectional_lstm(
        inputs=context_query_aware,
        input_lengths=context_length,
        size=50,
        name='context_query_aware_lstm'
    )

    spans = []
    for k, v in span2position.items():
        start = context_query_aware_lstm[:, k[0]]
        end = context_query_aware_lstm[:, k[1]]
        span = tf.concat([start, end], axis=-1, name='span_{}'.format(v))
        spans.append(span)

    spans = tf.stack(spans, axis=1, name='stacked_span_representations')

    logits = tf.layers.dense(
        inputs=spans,
        units=1,
        name='final_dense_layer'
    )

    return logits
